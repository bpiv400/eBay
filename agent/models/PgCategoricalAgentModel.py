import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import substitute_prefix
from constants import MODEL_NORM
from agent.agent_consts import PARAM_SHARING
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
    def __init__(self, init_dict_policy=None, init_dict_value=None,
                 sizes_policy=None, sizes_value=None, byr=False,
                 delay=False, dropout=(0., 0.)):
        """
        kwargs:
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super(PgCategoricalAgentModel, self).__init__(byr=byr,
                                                      delay=delay,
                                                      sizes=sizes_policy)
        self.policy_network = FeedForward(sizes=sizes_policy, dropout=dropout,
                                          norm=MODEL_NORM)
        self.value_network = FeedForward(sizes=sizes_value, dropout=dropout,
                                         norm=MODEL_NORM)
        with torch.no_grad():
            vals = np.arange(0, sizes_value['out'] / 100, 0.01)
            self.values = nn.Parameter(torch.from_numpy(vals), requires_grad=False)

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

    def _init_policy(self, init_dict=None):
        substitute_prefix(old_prefix='output.', new_prefix='output_action.',
                          state_dict=init_dict)
        if not PARAM_SHARING:
            substitute_prefix(old_prefix='nn1.', new_prefix='nn1_action.',
                              state_dict=init_dict)
        self._verify_init_dict(init_dict=init_dict)
        self.load_state_dict(state_dict=init_dict, strict=False)

    def _verify_init_dict(self, init_dict=None):
        """
        Verify that:
            1. Init dict contains no keys that are missing from
            the current agent module
            2. All of the parameters in the current agent module
            missing from init dict are related to the value function
        :param init_dict: dict of parameters used to initialize
        policy function
        :return: None
        """
        agent_keys = set(list(self.state_dict().keys()))
        init_keys = set(list(init_dict.keys()))

        # Statement 1.
        init_minus_agent = init_keys.difference(agent_keys)
        if len(init_minus_agent) > 0:
            msg = "Init dictionary contains unexpected params: {}".format(
                init_minus_agent)
            raise RuntimeError(msg)

        # Statement 2.
        agent_minus_init = agent_keys.difference(init_keys)
        unexpected_keys = list()
        value_suffix = '_value'
        for key in agent_minus_init:
            if value_suffix not in key:
                unexpected_keys.append(key)

        if len(unexpected_keys) > 0:
            msg = "Agent contains unexpected params: {}".format(unexpected_keys)
            raise RuntimeError(msg)
