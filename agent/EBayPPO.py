from collections import namedtuple
import numpy as np
import torch
from torch.optim import Adam
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from agent.const import LR_POLICY, LR_VALUE, RATIO_CLIP, STOP_ENTROPY, \
    FIELDS, PERIOD_EPOCHS

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs", "action", "return_", "advantage",
                              "valid", "pi_old"])
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
        self.entropy_coef = entropy
        self.entropy_step = self.entropy_coef / PERIOD_EPOCHS

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
        self._optim_value = Adam(self.agent.value_parameters(),
                                 lr=LR_VALUE, amsgrad=True)
        self._optim_policy = Adam(self.agent.policy_parameters(),
                                  lr=LR_POLICY, amsgrad=True)

    def optimize_agent(self, samples):
        """
        Train the agents, for multiple epochs over minibatches taken from the
        input samples.  Organizes agents inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if self.agent.recurrent or hasattr(self.agent, "update_obs_rms"):
            raise NotImplementedError()

        # initialize opt_info, compute valid obs and (discounted) return
        opt_info, valid, return_ = self.agent.process_samples(samples)

        # calculate advantage
        value = samples.agent.agent_info.value
        advantage = return_ - value
        opt_info['Value'] = value[valid].numpy()
        opt_info['Advantage'] = advantage[valid].numpy()

        # for getting policy and value in eval mode
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        # reshape inputs to loss function
        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
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
        policy_loss, value_loss, entropy = \
            self.loss(*loss_inputs[T_idxs, B_idxs])

        # policy step
        policy_loss.backward()
        self._optim_policy.step()

        # value step
        value_loss.backward()
        self._optim_value.step()

        # save for logging
        opt_info['Loss_Policy'] = policy_loss.item()
        opt_info['Loss_Value'] = value_loss.item()
        opt_info['Loss_EntropyBonus'] = self.entropy_coef
        entropy = entropy.detach().numpy()
        opt_info['Entropy'] = entropy

        # increment counter, reduce entropy bonus, and set complete flag
        self.update_counter += 1
        period = int(self.update_counter / PERIOD_EPOCHS)
        if period % 2 == 1:
            self.entropy_coef -= self.entropy_step
        if entropy.mean() < STOP_ENTROPY:
            self.training_complete = True

        # enclose in lists
        for k in FIELDS:
            if k not in opt_info:
                opt_info[k] = []
            else:
                opt_info[k] = [opt_info[k]]

        return OptInfo(**opt_info)

    def loss(self, agent_inputs, action, return_, advantage, valid, pi_old):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agents to compute forward pass on training data, and uses
        the ``agents.distribution`` to compute likelihoods and entropies.
        """
        # agents outputs
        pi_new, value_params = self.agent(*agent_inputs)

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
        value_loss = self.agent.get_value_loss(value_params, return_, valid)

        # entropy bonus
        entropy = self.agent.distribution.entropy(pi_new)[valid]
        entropy_loss = - self.entropy_coef * entropy.mean()

        # total loss
        policy_loss = pi_loss + entropy_loss

        # return loss values and statistics to record
        return policy_loss, value_loss, entropy

    def optim_state_dict(self):
        return {
            'value': self._optim_value.state_dict(),
            'policy': self._optim_policy.state_dict()
        }
