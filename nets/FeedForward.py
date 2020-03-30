import torch
import torch.nn as nn
from nets.nets_consts import BATCHNORM
from nets.nets_utils import FullyConnected, create_embedding_layers, create_groupings


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=(0.0, 0.0), norm='batch'):
        """
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param dropout: tuple of dropout rates.
        """
        super(FeedForward, self).__init__()

        # expand embeddings
        groups = create_groupings(sizes)

        self.nn0, total = create_embedding_layers(groups=groups,
                                                  sizes=sizes,
                                                  dropout=dropout[0],
                                                  norm=norm)

        # fully connected
        self.nn1 = FullyConnected(total, 
                                  sizes['out'],
                                  dropout=dropout[1],
                                  norm=norm)

    def forward(self, x):
        """
        x: OrderedDict()
        """
        elements = []
        for k in self.nn0.keys():
            elements.append(self.nn0[k](x))
        return self.nn1(torch.cat(elements, dim=1))



