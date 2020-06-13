import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from constants import MODEL_NORM
from util import load_state_dict, load_sizes
from agent.agent_utils import get_network_name
from agent.models.AgentModel import AgentModel
from nets.FeedForward import FeedForward


class PgCategoricalAgentModel(AgentModel):
    """
    Agent for eBay simulation

    1. Fully separate networks for value and policy networks
    2. Policy network outputs categorical probability distribution
     over actions
    3. Both networks use batch normalization
    4. Both networks use dropout and share dropout hyperparameters
    """
    def __init__(self, byr=False, delay=False, dropout=(0., 0.)):
        """
        kwargs:
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super(PgCategoricalAgentModel, self).__init__(byr=byr,
                                                      delay=delay)
        self.dropout = dropout
        self.policy_network = self._init_network(policy=True)
        self.value_network = self._init_network(policy=False)
        with torch.no_grad():
            vals = np.arange(0, self.value_network.sizes['out'] / 100, 0.01)
            self.values = nn.Parameter(torch.from_numpy(vals).float(),
                                       requires_grad=False)

    def value_parameters(self):
        return self.value_network.parameters()

    def policy_parameters(self):
        return self.policy_network.parameters()

    def zero_values_grad(self):
        if self.values.grad is not None:
            self.values.grad.detach_()
            self.values.grad.zero_()

    def con(self, input_dict=None):
        logits, _ = self._forward_dict(input_dict=input_dict, compute_value=False)
        logits = logits.squeeze()
        return logits

    def pi(self, observation, prev_action, prev_reward):
        input_dict = observation._asdict()
        logits = self.con(input_dict=input_dict)
        pi = softmax(logits, dim=logits.dim() - 1)
        return pi

    def _forward_dict(self, input_dict=None, compute_value=True):
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()
        pi_logits = self.policy_network(input_dict)
        # value output head
        if compute_value:
            v_logits = self.value_network(input_dict)
            value_distribution = softmax(v_logits, dim=v_logits.dim() - 1)
            v = torch.matmul(value_distribution, self.values)
        else:
            v = None

        return pi_logits, v

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        # setup input dictionary and fix dimensionality
        input_dict = observation._asdict()

        # temporary check
        # if self.training:
        #     torch.all(VALUE_STANDARD.eq(self.values.data))

        logits, v = self._forward_dict(input_dict=input_dict,
                                       compute_value=True)

        pi = softmax(logits, dim=logits.dim() - 1)

        # transformations
        pi = pi.squeeze()
        v = v.squeeze()

        return pi, v

    def _init_network(self, policy=True):
        network_name = get_network_name(byr=self.byr, policy=policy)
        sizes = load_sizes(network_name)
        init_dict = load_state_dict(network_name)
        net = FeedForward(sizes=sizes,
                          dropout=self.dropout,
                          norm=MODEL_NORM)
        net.load_state_dict(init_dict, strict=True)
        return net
