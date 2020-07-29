import numpy as np
import torch
from torch.optim import Adam
from collections import namedtuple
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from agent.models.HumanAgentModel import HumanAgentModel
from agent.Prefs import SellerPrefs, BuyerPrefs
from agent.const import STOPPING_EPOCHS, LR, RATIO_CLIP, ENTROPY_BONUS

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs",
                              "action",
                              "return_",
                              "advantage",
                              "valid",
                              "old_dist_info"])
OptInfo = namedtuple("OptInfo",
                     ["ActionsPerTraj",
                      "ConRate",
                      "Concession",
                      "DiscountedReturn",
                      "Advantage",
                      "Entropy",
                      "KL",
                      "PolicyLoss",
                      "ValueError"])


class EBayPPO:
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """
    mid_batch_reset = False
    bootstrap_value = True
    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(self, byr=None, delta=None, beta=None, kl=None):
        # save parameters to self
        self.kl_coeff = kl

        # agent preferences
        pref_cls = BuyerPrefs if byr else SellerPrefs
        self.prefs = pref_cls(delta=delta, beta=beta)

        # for cross-entropy
        self.human = HumanAgentModel(byr=byr).to('cuda')

        # parameters to be defined later
        self.agent = None
        self._optim_value = None
        self._optim_policy = None

        # count number of updates
        self.update_counter = 0

        # for stopping
        self.training_complete = False

    def initialize(self, agent=None):
        """
        Called by runner.
        """
        self.agent = agent

        # optimizers
        self._optim_value = Adam(self.agent.value_parameters(), lr=LR)
        self._optim_policy = Adam(self.agent.policy_parameters(), lr=LR)

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
        # valid[done.bool()] = 0.  # last obs of traj doesn't have action
        return valid.bool()

    def optimize_agent(self, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if self.agent.recurrent:
            raise NotImplementedError()

        # Move agent inputs to device once, index there
        env = samples.env
        agent_inputs = AgentInputs(
            observation=env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        if hasattr(self.agent, "update_obs_rms"):
            raise NotImplementedError()

        # break out samples
        reward, done, info = (env.reward, env.done, env.env_info)

        # time and/or action discounting
        return_, censored = self.prefs.discount_return(reward=reward,
                                                       done=done,
                                                       info=info)
        value = samples.agent.agent_info.value
        advantage = return_ - value

        # ignore steps from unfinished trajectories
        valid = self.valid_from_done(done)
        valid[censored] = False  # ignore censored actions

        # put sample components in LossInputs
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

        opt_info.ActionsPerTraj.append(info.num_actions[done].numpy())

        con = samples.agent.action[valid].numpy()
        con[con > 100] = 0
        con_rate = ((0 < con) & (con < 100)).mean().item()
        opt_info.ConRate.append(con_rate)
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
        policy_loss, value_error, entropy, kl = \
            self.loss(*loss_inputs[T_idxs, B_idxs])

        # policy step
        policy_loss.backward()
        self._optim_policy.step()

        # value step
        value_error.backward()
        self._optim_value.step()

        # save for logging
        opt_info.PolicyLoss.append(policy_loss.item())
        opt_info.ValueError.append(value_error.item())
        opt_info.Entropy.append(entropy)
        opt_info.KL.append(kl)

        # increment counter and set complete flag
        self.update_counter += 1
        if self.update_counter == STOPPING_EPOCHS:
            self.training_complete = True

        return opt_info

    def _get_entropy_loss(self, pi_new=None, valid=None):
        raise NotImplementedError()

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
        ratio = self.agent.distribution.likelihood_ratio(action,
                                                         old_dist_info=pi_old,
                                                         new_dist_info=pi_new)
        surr_1 = ratio * advantage
        clipped = torch.clamp(ratio, 1. - RATIO_CLIP, 1. + RATIO_CLIP)
        surr_2 = clipped * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        # loss from value estimation
        value_error = valid_mean(0.5 * (value - return_) ** 2, valid)

        # entropy
        entropy = self._entropy(p=pi_new.prob, valid=valid)

        # cross entropy
        pi_0 = self.human.get_policy(agent_inputs.observation).to('cpu')
        kl = self._cross_entropy(p=pi_0, q=pi_new.prob, valid=valid)

        # (cross-)entropy loss
        if self.kl_coeff is not None:
            entropy_loss = self.kl_coeff * kl.mean()
        else:
            entropy_loss = - ENTROPY_BONUS * entropy.mean()

        # total loss
        policy_loss = pi_loss + entropy_loss

        # loss values and statistics to record
        entropy = entropy.detach().numpy()
        kl = kl.detach().numpy()
        return policy_loss, value_error, entropy, kl

    def optim_state_dict(self):
        return {
            'value': self._optim_value.state_dict(),
            'policy': self._optim_policy.state_dict()
        }

    @staticmethod
    def _entropy(p=None, valid=None, eps=1e-8):
        entropy = -torch.sum(p * torch.log(p + eps), dim=-1)
        entropy = entropy[valid]
        return entropy

    @staticmethod
    def _cross_entropy(p=None, q=None, valid=None, eps=1e-8):
        kl = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)
        kl = kl[valid]
        return kl
