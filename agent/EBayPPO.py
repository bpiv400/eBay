import numpy as np
import torch
from torch.optim import Adam
from collections import namedtuple, OrderedDict
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from agent.util import define_con_set
from agent.const import PERIOD_EPOCHS, LR_POLICY, LR_VALUE, RATIO_CLIP, \
    ENTROPY_THRESHOLD, DEPTH_COEF, NOCON
from constants import IDX
from featnames import BYR, SLR

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs", "action", "return_", "advantage",
                              "turn", "valid", "pi_old"])
FIELDS = ["OffersPerTraj", "ThreadsPerTraj", "OffersPerThread", "DaysToDone",
          "Turn1_AccRate", "Turn1_RejRate", "Turn1_ConRate", "Turn1Con",
          "Turn2_AccRate", "Turn2_RejRate", "Turn2_ExpRate", "Turn2_ConRate", "Turn2Con",
          "Turn3_AccRate", "Turn3_RejRate", "Turn3_ConRate", "Turn3Con",
          "Turn4_AccRate", "Turn4_RejRate", "Turn4_ExpRate", "Turn4_ConRate", "Turn4Con",
          "Turn5_AccRate", "Turn5_RejRate", "Turn5_ConRate", "Turn5Con",
          "Turn6_AccRate", "Turn6_RejRate", "Turn6_ExpRate", "Turn6_ConRate", "Turn6Con",
          "Turn7_AccRate",
          "Rate_1", "Rate_2", "Rate_3", "Rate_4", "Rate_5", "Rate_6", "Rate_7", "Rate_Sale",
          "DollarReturn", "NormReturn", "Advantage", "Entropy",
          "Loss_Policy", "Loss_Value", "Loss_EntropyBonus", "Loss_DepthBonus"]
OptInfo = namedtuple("OptInfo", FIELDS)


class EBayPPO:
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agents.
    """
    mid_batch_reset = False
    bootstrap_value = True
    opt_info_fields = FIELDS

    def __init__(self, entropy=None):
        # save parameters to self
        self.entropy_coef = entropy
        self.depth_coef = DEPTH_COEF

        # parameters to be defined later
        self.agent = None
        self.byr = None
        self.con_set = None
        self._optim_value = None
        self._optim_policy = None

        # count number of updates
        self.update_counter = 0

        # for stopping
        self.entropy_step = self.entropy_coef / PERIOD_EPOCHS
        self.depth_step = self.depth_coef / PERIOD_EPOCHS
        self.training_complete = False

    def initialize(self, agent=None):
        """
        Called by runner.
        """
        self.agent = agent
        self.byr = agent.model.byr
        self.con_set = define_con_set(con_set=agent.model.con_set,
                                      byr=self.byr)

        # optimizers
        self._optim_value = Adam(self.agent.value_parameters(),
                                 lr=LR_VALUE, amsgrad=True)
        self._optim_policy = Adam(self.agent.policy_parameters(),
                                  lr=LR_POLICY, amsgrad=True)

    @staticmethod
    def discount_return(reward=None, done=None):
        """
        Computes time-discounted sum of future rewards from each
        time-step to the end of the batch. Sum resets where `done`
        is 1. Operations vectorized across all trailing dimensions
        after the first [T,].
        :param tensor reward: slr's normalized gross return.
        :param tensor done: indicator for end of trajectory.
        :return tensor return_: time-discounted return.
        """
        dtype = reward.dtype  # cast new tensors to this data type
        T, N = reward.shape  # time steps, number of envs

        # recast
        done = done.type(torch.int)

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        net_value = torch.zeros(N, dtype=dtype)

        for t in reversed(range(T)):
            net_value = net_value * (1 - done[t]) + reward[t] * done[t]
            return_[t] += net_value

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
        Train the agents, for multiple epochs over minibatches taken from the
        input samples.  Organizes agents inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if self.agent.recurrent or hasattr(self.agent, "update_obs_rms"):
            raise NotImplementedError()

        # break out samples
        env = samples.env
        reward, done, info = (env.reward, env.done, env.env_info)
        value = samples.agent.agent_info.value

        # ignore steps from unfinished trajectories
        valid = self.valid_from_done(done)

        # propagate return from end of trajectory (and discount, if necessary)
        return_ = self.discount_return(reward=reward, done=done)

        # initialize opt_info
        opt_info = OrderedDict()

        # various counts
        num_offers = info.num_offers[done].numpy()
        num_threads = info.num_threads[done].numpy()
        opt_info['OffersPerTraj'] = num_offers
        opt_info['ThreadsPerTraj'] = num_threads
        opt_info['OffersPerThread'] = num_offers / num_threads
        opt_info['DaysToDone'] = info.days[done].numpy()

        # action stats
        action = samples.agent.action[valid].numpy()
        con = np.take_along_axis(self.con_set, action, 0)
        turn = info.turn[valid].numpy()
        role = BYR if self.byr else SLR
        for t in IDX[role]:
            opt_info['Rate_{}'.format(t)] = np.mean(turn == t)

            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            if not self.byr:
                opt_info['{}_{}'.format(prefix, 'ExpRate')] = np.mean(con_t > 1)
            if t < 7:
                opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)
                if self.con_set != NOCON:
                    opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                        np.mean((con_t > 0) & (con_t < 1))
                    opt_info['{}{}'.format(prefix, 'Con')] = \
                        con_t[(con_t > 0) & (con_t < 1)]

        opt_info['Rate_Sale'] = np.mean(reward[done].numpy() > 0.)
        opt_info['DollarReturn'] = return_[done].numpy()
        opt_info['NormReturn'] = (return_[done] / info.max_return[done]).numpy()

        # normalize return and calculate advantage
        return_ /= info.max_return
        advantage = return_ - value
        opt_info['Advantage'] = advantage[valid].numpy()

        # for getting policy and value in eval mode
        agent_inputs = AgentInputs(
            observation=env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        # reshape inputs to loss function
        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            turn=info.turn,
            valid=valid,
            pi_old=samples.agent.agent_info.dist_info
        )

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
        opt_info['Loss_Policy'] = policy_loss.item()
        opt_info['Loss_Value'] = value_error.item()
        opt_info['Loss_EntropyBonus'] = self.entropy_coef
        opt_info['Loss_DepthBonus'] = self.depth_coef
        entropy = entropy.detach().numpy()
        opt_info['Entropy'] = entropy

        # increment counter, reduce entropy bonus, and set complete flag
        self.update_counter += 1
        if 2 * PERIOD_EPOCHS > self.update_counter >= PERIOD_EPOCHS:
            self.entropy_coef -= self.entropy_step
            self.depth_coef -= self.depth_step
        if entropy.mean() < ENTROPY_THRESHOLD:
            self.training_complete = True

        # enclose in lists
        for k in FIELDS:
            if k not in opt_info:
                opt_info[k] = []
            else:
                opt_info[k] = [opt_info[k]]

        return OptInfo(**opt_info)

    def loss(self, agent_inputs, action, return_, advantage, turn, valid, pi_old):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agents to compute forward pass on training data, and uses
        the ``agents.distribution`` to compute likelihoods and entropies.
        """
        # agents outputs
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

        # entropy bonus
        entropy = self.agent.distribution.entropy(pi_new)[valid]
        entropy_loss = - self.entropy_coef * entropy.mean()

        # depth bonus
        if self.depth_coef > 0:
            depth = (turn[valid] + 1) // 2 - 1
            depth_loss = - self.depth_coef * depth.float().mean()
        else:
            depth_loss = 0

        # total loss
        policy_loss = pi_loss + entropy_loss + depth_loss

        # return loss values and statistics to record
        return policy_loss, value_error, entropy

    def optim_state_dict(self):
        return {
            'value': self._optim_value.state_dict(),
            'policy': self._optim_policy.state_dict()
        }
