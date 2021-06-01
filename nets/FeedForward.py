import torch
import torch.nn as nn
from nets.const import HIDDEN_LARGE, HIDDEN_SMALL, HIDDEN_TINY
from nets.util import FullyConnected, create_embedding_layers, create_groupings
from featnames import META

MODEL_NORM = 'batch'  # normalization type


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=(0.0, 0.0)):
        """
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param tuple dropout: pair of dropout rates.
        """
        super().__init__()
        self.sizes = sizes

        # expand embeddings
        groups = create_groupings(sizes)

        self.nn0, total = create_embedding_layers(groups=groups,
                                                  sizes=sizes,
                                                  dropout=dropout[0],
                                                  norm=MODEL_NORM)

        # number of hidden nodes
        if META in sizes['x']:
            hidden = HIDDEN_LARGE
        elif 'offer1' in sizes['x']:
            hidden = HIDDEN_SMALL
        else:
            hidden = HIDDEN_TINY

        # fully connected
        self.nn1 = FullyConnected(total,
                                  hidden=hidden,
                                  dropout=dropout[1],
                                  norm=MODEL_NORM)

        self.output = nn.Linear(hidden, sizes['out'])

    def forward(self, x):
        """
        x: OrderedDict()
        """
        elements = []
        for k in self.nn0.keys():
            out = self.nn0[k](x)
            elements.append(out)
        x = self.nn1(torch.cat(elements, dim=elements[0].dim() - 1))
        return self.output(x)
