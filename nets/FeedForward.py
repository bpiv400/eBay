import torch, torch.nn as nn
from collections import OrderedDict
from nets.nets_utils import FullyConnected, Embedding


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

        # embeddings
        d, total = OrderedDict(), 0
        for name, group in groups.items():
            counts = {k: v for k, v in sizes['x'].items() if k in group}
            d[name] = Embedding(counts)
            total += sum(counts.values())
        self.nn0 = nn.ModuleDict(d)

        # fully connected
        self.nn1 = FullyConnected(total, sizes['out'], dropout=dropout)

    def forward(self, x):
        """
        x: OrderedDict()
        """
        elements = []
        for k in self.nn0.keys():
            elements.append(self.nn0[k](x))
        return self.nn1(torch.cat(elements, dim=1))


def create_groupings(sizes):
    groups = dict()
    groups['w2v'] = ['lstg', 'w2v_slr', 'w2v_byr']
    if 'slr' in sizes['x']:
        groups['other'] = ['lstg', 'cat', 'cndtn', 'slr']
    else:
        groups['other'] = ['lstg', 'cat', 'cndtn']
    if 'offer1' in sizes['x']:
        groups['offer'] = ['lstg'] \
            + [k for k in sizes['x'].keys() if 'offer' in k]
    return groups
