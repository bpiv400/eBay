import torch
from collections import namedtuple
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.algos.utils import discount_return, \
    generalized_advantage_estimation
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
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
                      "value_error",
                      "entropy",
                      "cross_entropy"])


class CrossEntropyPPO(PolicyGradientAlgo):
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
            normalize_advantage=False,
            mid_batch_reset=True
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

        # output fields
        self.opt_info_fields = tuple(f for f in OptInfo._fields)

        # parameters to be definied later
        self._batch_size = None
        self._ratio_clip = self.ratio_clip
        self._value_loss_coeff = self.value_loss_coeff
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

    @staticmethod
    def valid_from_done(done):
        """Returns a float mask which is zero for all time-steps after the last
        `done=True` is signaled.  This function operates on the leading dimension
        of `done`, assumed to correspond to time [T,...], other dimensions are
        preserved."""
        done = done.type(torch.float)
        done_count = torch.cumsum(done, dim=0)
        done_max, _ = torch.max(done_count, dim=0)
        valid = torch.abs(done_count - done_max) + done
        valid = torch.clamp(valid, max=1)
        return valid

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples. Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation. Masks out invalid samples.
        """
        print("Complete training iteration")
        reward, done, value, bv = (samples.env.reward,
                                   samples.env.done,
                                   samples.agent.agent_info.value,
                                   samples.agent.bootstrap_value)
        done = done.type(reward.dtype)

        # action-based discounting and GAE
        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        # zero out steps from unfinished trajectories
        valid = self.valid_from_done(done)

        # do not normalize advantage
        if self.normalize_advantage:
            raise NotImplementedError()

        return return_, advantage, valid

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
                loss, value_error, entropy, cross_entropy = self.loss(
                    *loss_inputs[T_idxs, B_idxs])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.value_error.append(value_error.item())
                opt_info.entropy.append(entropy.item())
                opt_info.cross_entropy.append(cross_entropy.item())
                self.update_counter += 1

        # step down learning rate
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * self.step(itr)

        # step down value error penalty
        self.value_loss_coeff = \
            self._value_loss_coeff * self.step(itr+1)

        # step down entropy bonus
        self.entropy_loss_coeff = \
            self._entropy_loss_coeff * self.step(itr+1)

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
        the ``agent.distribution`` to compute likelihoods and entropies.
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
        value_error = valid_mean(0.5 * (value - return_) ** 2, valid)
        value_loss = self.value_loss_coeff * value_error

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
        return loss, value_error, entropy, cross_entropy
