import torch
from torch.nn.utils import clip_grad_norm_
from torch import optim
import numpy as np
from collections import namedtuple
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.misc import iterate_mb_idxs
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from utils import slr_reward

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


class CrossEntropyPPO(RlAlgorithm):
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """

    def __init__(
            self,
            monthly_discount=0.995,
            action_discount=1,
            action_cost=0,
            learning_rate=0.001,
            value_loss_coeff=1.,
            cross_entropy_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            clip_grad_norm=1.,
            minibatches=4,
            ratio_clip=0.1
    ):
        save__init__args(locals())

        # output fields
        self.opt_info_fields = tuple(f for f in OptInfo._fields)

        # parameters to be definied later
        self._batch_size = None
        self.agent = None
        self.n_itr = None
        self.batch_spec = None
        self.optimizer = None
        self.lr_scheduler = None
        self.init_agent = None

        # for stepping down parameters later
        self._ratio_clip = self.ratio_clip
        self._entropy_loss_coeff = self.entropy_loss_coeff

    def step(self, itr):
        """
        Returns a multipicative factor that decreases linearly with itr.
        :param int itr: iteration
        :return float: coefficient on initial parameter value
        """
        return float((self.n_itr - itr) / self.n_itr)

    def initialize(self, *args, **kwargs):
        """
        Called by runner.
        """
        self.agent = kwargs['agent']
        self.n_itr = kwargs['n_itr']
        self.batch_spec = kwargs['batch_spec']

        # init_agent new initialized agent
        self.init_agent = PgCategoricalAgentModel(
            **self.agent.model_kwargs).to(self.agent.device)

        # for logging
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.

        # optimization
        self.optimizer = optim.Adam(self.agent.parameters(),
                                    lr=self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
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

    def discount_return(self, reward=None, done=None, months=None):
        """
        Computes time-discounted sum of future rewards from each
        time-step to the end of the batch. Sum resets where `done`
        is 1. Operations vectorized across all trailing dimensions
        after the first [T,].
        :param tensor reward: slr's normalized gross return.
        :param tensor done: indicator for end of trajectory.
        :param tensor months: months since beginning of listing.
        :return tensor return_: time-discounted return.
        """
        dtype = reward.dtype  # cast new tensors to this data type
        T, N = reward.shape  # time steps, number of environments

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        months_to_sale = torch.tensor(1e8, dtype=dtype).expand(N)
        action_diff = torch.zeros(N, dtype=dtype)
        sale_proceeds = torch.zeros(N, dtype=dtype)

        for t in reversed(range(T)):
            # update sale outcomes when sales are observed
            months_to_sale = months_to_sale * (1-done[t]) + months[t] * done[t]
            action_diff = (action_diff + 1) * (1-done[t])
            sale_proceeds = sale_proceeds * (1-done[t]) + reward[t] * done[t]

            # discounted sale proceeds
            return_[t] += slr_reward(months_to_sale=months_to_sale,
                                     months_since_start=months[t],
                                     sale_proceeds=sale_proceeds,
                                     monthly_discount=self.monthly_discount,
                                     action_diff=action_diff,
                                     action_discount=self.action_discount,
                                     action_cost=self.action_cost)

        return return_

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples. Discounts returns and masks out invalid samples.
        """
        # break out samples
        env = samples.env
        reward, done, months, start_price = (env.reward,
                                             env.done,
                                             env.env_info.months,
                                             env.env_info.start_price)
        done = done.type(reward.dtype)
        months = months.type(reward.dtype)
        start_price = start_price.type(reward.dtype)

        # time discounting
        return_ = self.discount_return(reward=reward,
                                       done=done,
                                       months=months)
        return_ /= start_price  # normalize
        advantage = return_ - samples.agent.agent_info.value
        
        # zero out steps from unfinished trajectories
        valid = self.valid_from_done(done)

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

        # loop over minibatches
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        batch_size = T * B
        mb_size = batch_size // self.minibatches
        for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
            T_idxs = idxs % T
            B_idxs = idxs // T
            self.optimizer.zero_grad()
            loss, value_error, entropy, cross_entropy = self.loss(
                *loss_inputs[T_idxs, B_idxs])
            loss.backward()
            grad_norm = clip_grad_norm_(self.agent.parameters(),
                                        self.clip_grad_norm)
            self.optimizer.step()

            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.value_error.append(value_error.item())
            opt_info.entropy.append(entropy.item())
            opt_info.cross_entropy.append(cross_entropy.item())
            self.update_counter += 1

        # step down learning rate
        self.lr_scheduler.step()
        self.ratio_clip = self._ratio_clip * self.step(itr)

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
