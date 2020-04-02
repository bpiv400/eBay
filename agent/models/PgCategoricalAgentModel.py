import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import substitute_prefix
from agent.agent_consts import PARAM_SHARING
from nets.nets_utils import (create_embedding_layers, create_groupings,
                             FullyConnected)
from nets.nets_consts import HIDDEN


class PgCategoricalAgentModel(nn.Module):
    """
    Returns a vector of
    """
    def __init__(self, init_dict=None, sizes=None, norm=None):
        """
        kwargs:
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super(PgCategoricalAgentModel, self).__init__()
        # expand embeddings
        groups = create_groupings(sizes=sizes)
        # create embedding layers
        self.nn0, total = create_embedding_layers(groups=groups, sizes=sizes,
                                                  dropout=0.0, norm=norm)

        if PARAM_SHARING:
            # shared fully connected layers
            self.nn1 = FullyConnected(total, dropout=0.0, norm=norm)
            self.nn1_action = None
            self.nn1_value = None
        else:
            self.nn1_action = FullyConnected(total, dropout=0.0, norm=norm)
            self.nn1_value = FullyConnected(total, dropout=0.0, norm=norm)
            self.nn1 = None
        # output layers
        self.output_value = nn.Linear(HIDDEN, 1)
        self.output_action = nn.Linear(HIDDEN, sizes['out'])

        if init_dict is not None:
            self._init_policy(init_dict=init_dict)

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        x = observation._asdict()
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        embedded = torch.cat(l, dim=l[0].dim() - 1)

        # fully connected
        if PARAM_SHARING:
            hidden_action = self.nn1(embedded)
            hidden_value = hidden_action
        else:
            hidden_action = self.nn1_action(embedded)
            hidden_value = self.nn1_value(embedded)

        # value output head
        v = self.output_value(hidden_value)

        # policy output head
        logits = self.output_action(hidden_action)
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