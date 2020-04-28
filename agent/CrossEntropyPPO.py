import torch
from collections import namedtuple
from rlpyt.algos.pg.ppo import PPO
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs",
                              "action",
                              "return_",
                              "advantage",
                              "valid",
                              "old_dist_info"])
OptInfo = namedtuple("OptInfo",
                     ["loss",
                      "gradNorm",
                      "entropy",
                      "cross_entropy"])


class CrossEntropyPPO(PPO):
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            cross_entropy_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

        # parameters to be definied later
        self._batch_size = None
        self._ratio_clip = self.ratio_clip
        self._entropy_loss_coeff = self.entropy_loss_coeff
        self.lr_scheduler = None
        self.init_agent = None

    def step(self, itr):
        """
        Returns a multipicative factor that decreases linearly with itr.
        :param int itr: iteration
        :return float: coefficient on initial parameter value
        """
        return float((self.n_itr - itr) / self.n_itr)

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)

        # init_agent new initialized agent
        self.init_agent = PgCategoricalAgentModel(
            **self.agent.model_kwargs).to(self.agent.device)

        # original PPO initialization
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer, lr_lambda=self.step)

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if self.agent.recurrent:
            raise NotImplementedError()

        # Move agent inputs to device once, index there.
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)

        # extract sample components and put them in LossInputs
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        # loop over algorithm epochs
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        batch_size = T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = idxs % T
                B_idxs = idxs // T
                self.optimizer.zero_grad()
                loss, entropy, cross_entropy = self.loss(
                    *loss_inputs[T_idxs, B_idxs])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.cross_entropy.append(cross_entropy.item())
                self.update_counter += 1

        # step down learning rate
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * self.step(itr)

        # step down entropy bonus
        self.entropy_loss_coeff = self._entropy_loss_coeff * self.step(itr)

        return opt_info

    @staticmethod
    def mean_kl(p, q, valid=None, eps=1e-8):
        kl = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)
        return valid_mean(kl, valid)

    def loss(self, agent_inputs, action, return_, advantage, valid, pi_old):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        # agent's policy
        pi_new, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        # loss from policy
        ratio = dist.likelihood_ratio(action,
                                      old_dist_info=pi_old,
                                      new_dist_info=pi_new)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
                                    1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        # loss from value estimation
        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # cross-entropy loss
        pi_0, _ = self.init_agent(*agent_inputs)
        cross_entropy = self.mean_kl(pi_0.to('cpu'), pi_new.prob, valid)
        cross_entropy_loss = self.cross_entropy_loss_coeff * cross_entropy

        # entropy bonus
        entropy = dist.mean_entropy(pi_new, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        # total loss
        loss = pi_loss + value_loss + cross_entropy_loss + entropy_loss

        # cross-entropy replaces entropy
        return loss, entropy, cross_entropy
