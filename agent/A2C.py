from collections import namedtuple
import numpy as np
import torch
from torch.nn.functional import huber_loss
from torch.optim import Adam
from rlpyt.utils.collections import namedarraytuple
from agent.const import LR, STOP_ENTROPY, FIELDS, PERIOD_EPOCHS, ENTROPY_COEF, MAX_GRAD_NORM

LossInputs = namedarraytuple("LossInputs",
                             ["observation", "action", "return_", "advantage", "valid"])
OptInfo = namedtuple("OptInfo", FIELDS)


class A2C:
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agents.
    """
    mid_batch_reset = False
    bootstrap_value = True
    opt_info_fields = FIELDS

    def __init__(self):
        self.entropy_coef = ENTROPY_COEF
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
                                 lr=LR, amsgrad=True)
        self._optim_policy = Adam(self.agent.policy_parameters(),
                                  lr=LR, amsgrad=True)

    def optimize_agent(self, samples):
        """
        Train the agents, for multiple epochs over minibatches taken from the
        input samples.  Organizes agents inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        # initialize opt_info, compute valid obs and (discounted) return
        opt_info, valid, return_ = self.agent.process_samples(samples)

        # calculate advantage
        value = samples.agent.agent_info.value
        advantage = return_ - value
        opt_info['Value'] = value[valid].numpy()
        opt_info['Advantage'] = advantage[valid].numpy()

        # reshape inputs to loss function
        loss_inputs = LossInputs(
            observation=samples.env.observation,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid
        )

        # zero gradients
        self._optim_value.zero_grad()
        self._optim_policy.zero_grad()

        # loss/error
        t, b = samples.env.reward.shape[:2]
        idxs = np.arange(t * b)
        policy_loss, value_loss, entropy = \
            self.loss(*loss_inputs[idxs % t, idxs // t])

        # policy step
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.policy_parameters(), MAX_GRAD_NORM)
        self._optim_policy.step()

        # value step
        value_loss.backward()
        self._optim_value.step()

        # save for logging
        opt_info['Loss_Policy'] = policy_loss.item()
        opt_info['Loss_GradNorm'] = grad_norm.item()
        opt_info['Loss_Value'] = value_loss.item()
        opt_info['Loss_EntropyBonus'] = self.entropy_coef
        opt_info['Entropy'] = entropy.detach().numpy()

        # increment counter, reduce entropy bonus, and set complete flag
        self.update_counter += 1
        period = int(self.update_counter / PERIOD_EPOCHS)
        if period == 1:
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

    def loss(self, observation, action, return_, advantage, valid):
        """
        Computes A2C loss.
        """
        # agents outputs
        pi, v = self.agent(observation)

        # loss from policy
        lnl = self.agent.distribution.log_likelihood(action, pi)
        assert lnl.shape == advantage.shape
        surrogate = lnl * advantage
        pi_loss = - surrogate[valid].mean()

        # loss from value estimation
        assert v.shape == return_.shape
        value_loss = huber_loss(v, return_.float(), reduction='none')[valid].mean()

        # entropy bonus
        entropy = self.agent.distribution.entropy(pi)[valid]
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
