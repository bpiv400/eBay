from collections import OrderedDict
import torch
import torch.nn as nn
from agent.const import AGENT_HIDDEN
from nets.const import HIDDEN
from nets.util import FullyConnected, create_embedding_layers, create_groupings
from constants import MODEL_NORM


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=(0.0, 0.0), agent=False, norm=MODEL_NORM):
        """
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param tuple dropout: pair of dropout rates.
        :param bool agent: uses smaller model if True
        :param str norm: type of normalization.
        """
        super().__init__()
        self.sizes = sizes

        # expand embeddings
        groups = create_groupings(sizes)

        self.nn0, total = create_embedding_layers(groups=groups,
                                                  sizes=sizes,
                                                  dropout=dropout[0],
                                                  norm=norm)

        # fully connected
        hidden = AGENT_HIDDEN if agent else HIDDEN
        self.nn1 = FullyConnected(total,
                                  hidden=hidden,
                                  dropout=dropout[1],
                                  norm=norm)

        self.output = nn.Linear(hidden, sizes['out'])

    def forward(self, x):
        """
        x: OrderedDict()
        """
        # convert tensor to dictionary, if necessary
        if type(x) is torch.Tensor:
            d = OrderedDict()
            ct = 0
            for k, v in self.sizes['x'].items():
                d[k] = x[:, ct:ct+v]
                ct += v
            assert x.size()[-1] == ct
            x = d

        # pass through layers
        elements = []
        for k in self.nn0.keys():
            out = self.nn0[k](x)
            elements.append(out)
        hidden = self.nn1(torch.cat(elements, dim=elements[0].dim() - 1))
        return self.output(hidden)
