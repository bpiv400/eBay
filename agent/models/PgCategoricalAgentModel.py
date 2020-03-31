import torch
import torch.nn as nn
from torch.nn.functional import softmax
from constants import SLR_INIT
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
        self.v_output = nn.Linear(HIDDEN, 1)
        self.pi_output = nn.Linear(HIDDEN, sizes['out'])

        self._init_policy(init_dict=init_dict)

    @staticmethod
    def detect_norm(init_dict=None):
        return "weight"

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
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
        pass
