import torch, torch.nn as nn
from collections import OrderedDict
from nets.nets_utils import FullyConnected, Embedding
from constants import EMBEDDING_GROUPS


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
        groups = EMBEDDING_GROUPS.copy()
        if 'offer1' in sizes['x']:
            groups['offer'] = ['lstg'] \
                + [k for k in sizes['x'].keys() if 'offer' in k]

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
        '''
        x: OrderedDict()
        '''
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        return self.nn1(torch.cat(l, dim=1))
