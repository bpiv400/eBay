import torch
import torch.nn as nn
from nets.nets_consts import BATCHNORM
from nets.nets_utils import FullyConnected, create_embedding_layers, create_groupings


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=False):
        """
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param dropout: boolean for Variational dropout in FullyConnected.
        """
        super(FeedForward, self).__init__()

        # save dropout boolean to self
        self.dropout = dropout

        # expand embeddings
        groups = create_groupings(sizes)

        self.nn0, total = create_embedding_layers(groups=groups,
                                                  sizes=sizes,
                                                  batch_norm=BATCHNORM)

        # fully connected
        self.nn1 = FullyConnected(total, sizes['out'], dropout=dropout,
                                  batch_norm=BATCHNORM)

    def forward(self, x):
        """
        x: OrderedDict()
        """
        elements = []
        for k in self.nn0.keys():
            elements.append(self.nn0[k](x))
        return self.nn1(torch.cat(elements, dim=1))



