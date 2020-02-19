import torch, torch.nn as nn
from collections import OrderedDict
from nets.nets_consts import BATCHNORM
from nets.nets_utils import FullyConnected, Embedding


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=True):
        '''
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param dropout: boolean for Variational dropout in FullyConnected.
        '''
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
        '''
        x: OrderedDict()
        '''
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        return self.nn1(torch.cat(l, dim=1))


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


def create_embedding_layers(groups=None, sizes=None, batch_norm=False):
    # embeddings
    d, total = OrderedDict(), 0
    for name, group in groups.items():
        counts = {k: v for k, v in sizes['x'].items() if k in group}
        d[name] = Embedding(counts, batch_norm=batch_norm)
        total += sum(counts.values())
    return nn.ModuleDict(d), total
