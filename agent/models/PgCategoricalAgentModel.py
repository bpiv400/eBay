import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from constants import MODEL_NORM
from utils import load_state_dict, load_sizes
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
            self.values = nn.Parameter(torch.from_numpy(vals), requires_grad=False)

    @property
    def value_params(self):
        return self.value_network.parameters()

    @property
    def policy_params(self):
        return self.policy_network.parameters()

    def con(self, input_dict=None):
        logits, _ = self._forward_dict(input_dict=input_dict, compute_value=False)
        logits = logits.squeeze()
        return logits

    def _forward_dict(self, input_dict=None, compute_value=True):
        logits = self.policy_network(input_dict)
        # value output head
        if compute_value:
            value_distribution = self.value_network(input_dict)
            v = torch.matmul(value_distribution, self.values)
        else:
            v = None

        return logits, v

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        input_dict = observation._asdict()
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
        net = FeedForward(sizes=sizes, dropout=self.dropout, norm=MODEL_NORM)
        net.load_state_dict(init_dict, strict=True)
        return net
