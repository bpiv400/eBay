import torch
import torch.nn as nn
from torch.nn.functional import softmax
from constants import SLR_INIT
from utils import substitute_prefix
from agent.agent_consts import PARAM_SHARING
from agent.agent_utils import load_init_model
from nets.nets_utils import (create_embedding_layers, create_groupings,
                             FullyConnected)
from nets.nets_consts import HIDDEN


class PgCategoricalAgentModel(nn.Module):
    """
    Returns a vector of
    """
    def __init__(self, init_model=SLR_INIT, sizes=None):
        """
        kwargs:
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super(PgCategoricalAgentModel, self).__init__()
        # load initialization model
        init_dict = load_init_model(name=init_model, size=sizes['out'])
        norm = self.detect_norm(init_dict=init_dict)
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

        self._init_policy(init_dict=init_dict)

    @staticmethod
    def detect_norm(init_dict=None):
        found_v, found_g = False, False
        for param_name in init_dict.keys():
            if '_v' in param_name:
                found_v = True
            elif '_g' in param_name:
                found_g = True
        if found_g and found_v:
            return "weight"
        else:
            raise NotImplementedError("Unexpected normalization type")

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v

        # TODO: FIX FUNCTION
        """
        x = observation._asdict()
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        embedded = torch.cat(l, dim=l[0].dim() - 1)
        v = self.nn1_value(embedded)
        logits = self.nn1_action(embedded)
        # apply softmax
        pi = softmax(logits, dim=logits.dim() - 1)
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