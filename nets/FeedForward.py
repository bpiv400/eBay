import torch
import torch.nn as nn
from nets.const import HIDDEN
from nets.util import FullyConnected, create_embedding_layers, create_groupings
from constants import MODEL_NORM


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=(0.0, 0.0), hidden=HIDDEN, norm=MODEL_NORM):
        """
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param dropout: tuple of dropout rates.
        :param hidden: size of each hidden layer.
        :param norm: string type of normalization.
        """
        super(FeedForward, self).__init__()
        self.sizes = sizes

        # expand embeddings
        groups = create_groupings(sizes)

        self.nn0, total = create_embedding_layers(groups=groups,
                                                  sizes=sizes,
                                                  dropout=dropout[0],
                                                  norm=norm)

        # fully connected
        self.nn1 = FullyConnected(total,
                                  hidden=hidden,
                                  dropout=dropout[1],
                                  norm=norm)

        self.output = nn.Linear(hidden, sizes['out'])

    def forward(self, x):
        """
        x: OrderedDict()
        """
        elements = []
        for k in self.nn0.keys():
            out = self.nn0[k](x)
            elements.append(out)
        hidden = self.nn1(torch.cat(elements, dim=elements[0].dim() - 1))
        return self.output(hidden)
