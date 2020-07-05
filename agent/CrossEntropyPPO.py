import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from collections import namedtuple
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.base import RlAlgorithm
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from constants import LR1, LR_FACTOR
from utils import slr_reward, max_slr_reward

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs",
                              "action",
                              "return_",
                              "advantage",
                              "valid",
                              "old_dist_info"])
OptInfo = namedtuple("OptInfo",
                     ["DiscountedReturn",
                      "Advantage",
                      "Entropy",
                      "GradNormPolicy",
                      "GradNormValue",
                      "PolicyLoss",
                      "ValueError",
                      "PolicyLR"])


class CrossEntropyPPO(RlAlgorithm):
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """

    def __init__(
            self,
            action_discount=1.,
            action_cost=0.,
            lr=0.001,
            same_lr=True,
            entropy_coeff=0.01,
            clip_grad_norm=1.,
            mb_size=128,
            ratio_clip=0.1,
            lr_step_batches=1,
            use_cross_entropy=False,
            debug_ppo=False
    ):
        # save parameters to self
        self.action_discount = action_discount
        self.action_cost = action_cost
        self.lr = lr
        self.same_lr = same_lr
        self.entropy_coeff = entropy_coeff
        self.clip_grad_norm = clip_grad_norm
        self.mb_size = mb_size
        self.ratio_clip = ratio_clip
        self.lr_step_batches = lr_step_batches
        self.use_cross_entropy = use_cross_entropy
        self.debug = debug_ppo
        self.loss_history = []  # for stepping down lr

        # output fields
        self.opt_info_fields = tuple(f for f in OptInfo._fields)

        # parameters to be defined later
        self.agent = None
        self.batch_spec = None
        self.lr_scheduler = None
        self.optimizer_value = None
        self.optimizer_policy = None
        if self.use_cross_entropy:
            self.init_agent = None

    def initialize(self, *args, **kwargs):
        """
        Called by runner.
        """
        self.agent = kwargs['agent']
        self.batch_spec = kwargs['batch_spec']

        # optimizers
        self.optimizer_value = Adam(self.agent.value_parameters(),
                                    lr=self.lr)
        self.optimizer_policy = Adam(self.agent.policy_parameters(),
                                     lr=self.lr)

        # init_agent new initialized agent
        if self.use_cross_entropy:
            self.init_agent = PgCategoricalAgentModel(
                **self.agent.model_kwargs).to(self.agent.device)

        # lr scheduler
        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer_policy,
                                              factor=LR_FACTOR,
                                              threshold=0.,
                                              patience=0)

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

    def discount_return(self, reward=None, done=None, months=None,
                        bin_proceeds=None):
        """
        Computes time-discounted sum of future rewards from each
        time-step to the end of the batch. Sum resets where `done`
        is 1. Operations vectorized across all trailing dimensions
        after the first [T,].
        :param tensor reward: slr's normalized gross return.
        :param tensor done: indicator for end of trajectory.
        :param tensor months: months since beginning of listing.
        :param tensor bin_proceeds: what seller would net if
        item sold for bin price immediately.
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
                                     action_diff=action_diff,
                                     action_discount=self.action_discount,
                                     action_cost=self.action_cost)

            # normalize
            max_return = max_slr_reward(months_since_start=months[t],
                                        bin_proceeds=bin_proceeds[t])
            return_[t] /= max_return

        return return_

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples. Discounts returns and masks out invalid samples.
        """
        # break out samples
        env = samples.env
        reward, done, months, bin_proceeds = (env.reward,
                                              env.done,
                                              env.env_info.months,
                                              env.env_info.bin_proceeds)
        done = done.type(reward.dtype)
        months = months.type(reward.dtype)
        bin_proceeds = bin_proceeds.type(reward.dtype)
        # print('reward shape: {}'.format(reward.shape))
        # time discounting
        return_ = self.discount_return(reward=reward,
                                       done=done,
                                       months=months,
                                       bin_proceeds=bin_proceeds)
        # print('return shape: {}'.format(return_.shape))
        # print('value shape: {}'.format(samples.agent.agent_info.value.shape))
        value = samples.agent.agent_info.value
        if self.debug:
            value = value[:, :, 0]
        advantage = return_ - value
        
        # zero out steps from unfinished trajectories
        valid = self.valid_from_done(done)

        return return_, advantage, valid

    def optimize_agent(self, samples):
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
        # for slicing
        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        # initialize opt_info with return and advantange
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        opt_info.DiscountedReturn.append(
            return_[valid.bool()].numpy())
        opt_info.Advantage.append(
            advantage[valid.bool()].numpy())

        # loop over minibatches
        T, B = samples.env.reward.shape[:2]
        batch_size = T * B
        total_policy_loss, total_value_error = 0., 0.
        # loop over minibatches
        for idxs in iterate_mb_idxs(batch_size, self.mb_size, shuffle=True):
            T_idxs = idxs % T
            B_idxs = idxs // T
            self.optimizer_value.zero_grad()
            self.optimizer_policy.zero_grad()
            # self.agent.zero_values_grad()

            # loss/error
            policy_loss, value_error, entropy = self.loss(
                *loss_inputs[T_idxs, B_idxs])
            total_policy_loss += policy_loss.item()
            total_value_error += value_error.item()

            # policy step
            policy_loss.backward()
            grad_norm_policy = clip_grad_norm_(
                self.agent.policy_parameters(),
                self.clip_grad_norm)
            self.optimizer_policy.step()

            # value step
            value_error.backward()
            grad_norm_value = clip_grad_norm_(
                self.agent.value_parameters(),
                self.clip_grad_norm)
            self.optimizer_value.step()

            # save for logging
            opt_info.Entropy.append(entropy)
            opt_info.GradNormPolicy.append(grad_norm_policy)
            opt_info.GradNormValue.append(grad_norm_value)

            # increment counter
            self.update_counter += 1

        # save loss/error and learning rate
        opt_info.PolicyLoss.append(total_policy_loss)
        opt_info.ValueError.append(total_value_error)

        # save learning rate
        lr = self.learning_rate
        opt_info.PolicyLR.append(lr)

        # step down learning rate
        self.loss_history.append(total_policy_loss)
        if len(self.loss_history) >= self.lr_step_batches:
            avg_loss = np.mean(self.loss_history[-self.lr_step_batches:])
            self.lr_scheduler.step(avg_loss)

            # clear loss history after stepping down learning rate
            if lr != self.learning_rate:
                self.loss_history = []

            # ensure value optimizer has same learning rate
            if self.same_lr:
                lr = self.learning_rate
                for param_group in self.optimizer_value.param_groups:
                    param_group['lr'] = lr

        return opt_info

    @property
    def learning_rate(self):
        for param_group in self.optimizer_policy.param_groups:
            return param_group['lr']

    @property
    def training_complete(self):
        return self.learning_rate < LR1

    @staticmethod
    def mean_kl(p, q, valid=None, eps=1e-8):
        kl = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)
        return valid_mean(kl, valid)

    @staticmethod
    def entropy(p, eps=1e-8):
        return -torch.sum(p * torch.log(p + eps), dim=-1)

    def loss(self, agent_inputs, action, return_, advantage, valid, pi_old):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.
        """
        # agent outputs
        pi_new, value = self.agent(*agent_inputs)

        # loss from policy
        dist = self.agent.distribution
        if self.debug:
            pi_new = buffer_to(DistInfo(prob=pi_new.prob.unsqueeze(1)),
                               device="cpu")

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

        # calculate entropy for each action
        entropy = self.entropy(pi_new.prob)[valid.bool()]

        # cross-entropy loss
        if self.use_cross_entropy:
            pi_0 = self.init_agent.con(*agent_inputs)
            cross_entropy = self.mean_kl(pi_0.to('cpu'), pi_new.prob, valid)
            entropy_loss = self.entropy_coeff * cross_entropy

        # entropy bonus
        else:
            entropy_loss = - self.entropy_coeff * entropy.mean()

        # total loss
        policy_loss = pi_loss + entropy_loss

        # loss values and statistics to record
        return policy_loss, value_error, entropy.detach().numpy()

    def optim_state_dict(self):
        return {
            'value': self.optimizer_value.state_dict(),
            'policy': self.optimizer_policy.state_dict()
        }
