import numpy as np
import torch
from torch.optim import Adam
from collections import namedtuple
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from agent.const import PERIOD_EPOCHS, LR_POLICY, LR_VALUE, RATIO_CLIP

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs",
                              "action",
                              "return_",
                              "advantage",
                              "valid",
                              "old_dist_info"])
OptInfo = namedtuple("OptInfo",
                     ["ActionsPerTraj",
                      "DelaysPerTraj",
                      "OffersPerTraj",
                      "MonthsToDone",
                      "Rate_Con",
                      "Rate_Acc",
                      "Rate_Rej",
                      "Rate_Exp",
                      "Rate_Sale",
                      "Concession",
                      "DiscountedReturn",
                      "Advantage",
                      "Entropy",
                      "Loss_Policy",
                      "Loss_Value",
                      "Loss_EntropyBonus"])


class EBayPPO:
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """
    mid_batch_reset = False
    bootstrap_value = True
    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(self, entropy_coeff=None):
        # save parameters to self
        self.entropy_coeff = entropy_coeff

        # parameters to be defined later
        self.agent = None
        self.byr = None
        self._optim_value = None
        self._optim_policy = None

        # count number of updates
        self.update_counter = 0

        # for stopping
        self.entropy_step = entropy_coeff / PERIOD_EPOCHS
        self.training_complete = False

    def initialize(self, agent=None):
        """
        Called by runner.
        """
        self.agent = agent
        self.byr = agent.model.byr

        # optimizers
        self._optim_value = Adam(self.agent.value_parameters(),
                                 lr=LR_VALUE, amsgrad=True)
        self._optim_policy = Adam(self.agent.policy_parameters(),
                                  lr=LR_POLICY, amsgrad=True)

    @staticmethod
    def discount_return(reward=None, done=None, info=None):
        """
        Computes time-discounted sum of future rewards from each
        time-step to the end of the batch. Sum resets where `done`
        is 1. Operations vectorized across all trailing dimensions
        after the first [T,].
        :param tensor reward: slr's normalized gross return.
        :param tensor done: indicator for end of trajectory.
        :param namedtuple info: contains other required feats
        :return tensor return_: time-discounted return.
        """
        dtype = reward.dtype  # cast new tensors to this data type
        T, N = reward.shape  # time steps, number of environments

        # recast
        done = done.type(torch.int)
        max_return = info.max_return.type(dtype)

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        net_value = torch.zeros(N, dtype=dtype)

        for t in reversed(range(T)):
            # value less price paid
            net_value = net_value * (1 - done[t]) + reward[t] * done[t]
            net_value = torch.min(net_value, max_return[t])  # precision
            return_[t] += net_value

        # normalize
        return_ /= max_return

        return return_

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
        return valid.bool()

    def optimize_agent(self, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if self.agent.recurrent or hasattr(self.agent, "update_obs_rms"):
            raise NotImplementedError()

        # break out samples
        env = samples.env
        reward, done, info = (env.reward, env.done, env.env_info)

        # ignore steps from unfinished trajectories
        valid = self.valid_from_done(done)

        # time and/or action discounting
        return_ = self.discount_return(reward=reward, done=done, info=info)

        # advantage
        value = samples.agent.agent_info.value
        advantage = return_ - value

        # Move inputs to device once, index there
        agent_inputs = AgentInputs(
            observation=env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        # initialize opt_info
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        if self.byr:
            opt_info.DelaysPerTraj.append(info.num_delays[done].numpy())
        opt_info.OffersPerTraj.append(info.num_offers[done].numpy())
        opt_info.MonthsToDone.append(info.months[done].numpy())

        con = samples.agent.action[valid].numpy()
        opt_info.Rate_Con.append(((0 < con) & (con < 100)).mean())
        opt_info.Rate_Acc.append((con == 100).mean())
        opt_info.Rate_Rej.append((con == 0).mean())
        opt_info.Rate_Sale.append((reward[done].numpy() > 0.).mean())
        if not self.byr:
            opt_info.Rate_Exp.append((con == 101).mean())

        con = con[(con < 100) & (con > 0)]
        opt_info.Concession.append(con)

        opt_info.DiscountedReturn.append(return_[valid].numpy())
        opt_info.Advantage.append(advantage[valid].numpy())

        # zero gradients
        self._optim_value.zero_grad()
        self._optim_policy.zero_grad()

        # loss/error
        T, B = samples.env.reward.shape[:2]
        idxs = np.arange(T * B)
        T_idxs = idxs % T
        B_idxs = idxs // T
        policy_loss, value_error, entropy = \
            self.loss(*loss_inputs[T_idxs, B_idxs])

        # policy step
        policy_loss.backward()
        self._optim_policy.step()

        # value step
        value_error.backward()
        self._optim_value.step()

        # save for logging
        opt_info.Loss_Policy.append(policy_loss.item())
        opt_info.Loss_Value.append(value_error.item())
        opt_info.Loss_EntropyBonus.append(self.entropy_coeff)
        opt_info.Entropy.append(entropy.detach().numpy())

        # increment counter, reduce entropy bonus, and set complete flag
        self.update_counter += 1
        if self.entropy_coeff > 0. and self.update_counter >= PERIOD_EPOCHS:
            self.entropy_coeff -= self.entropy_step
        if self.update_counter == 3 * PERIOD_EPOCHS:
            self.training_complete = True

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, pi_old):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.
        """
        # agent outputs
        pi_new, v = self.agent(*agent_inputs)

        # loss from policy
        ratio = self.agent.distribution.likelihood_ratio(action,
                                                         old_dist_info=pi_old,
                                                         new_dist_info=pi_new)
        clipped = torch.clamp(ratio, 1. - RATIO_CLIP, 1. + RATIO_CLIP)
        surr_1 = ratio * advantage
        surr_2 = clipped * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        # loss from value estimation
        value_error = valid_mean(0.5 * (v - return_) ** 2, valid)

        # entropy
        entropy = self.agent.distribution.entropy(pi_new)[valid]
        entropy_loss = - self.entropy_coeff * entropy.mean()

        # total loss
        policy_loss = pi_loss + entropy_loss

        # return loss values and statistics to record
        return policy_loss, value_error, entropy

    def optim_state_dict(self):
        return {
            'value': self._optim_value.state_dict(),
            'policy': self._optim_policy.state_dict()
        }
