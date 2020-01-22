import torch, torch.nn as nn
from collections import OrderedDict
from model.VariationalDropout import VariationalDropout
from constants import EMBEDDING_GROUPS


class Layer(nn.Module):
    def __init__(self, N_in, N_out, 
        batchnorm=True, affine=True, dropout=False):
        '''
        :param N_in: scalar number of input weights.
        :param N_out: scalar number of output weights.
        :param batchnorm: boolean for including batch normalization in each layer.
        :param affine: boolean for estimating affine weights in batch normalization.
        :param dropout: boolean for including variational dropout in each layer.
        '''
        super(Layer, self).__init__()
        # initialize layer
        self.layer = nn.ModuleList([nn.Linear(N_in, N_out)])

        # batch normalization
        if batchnorm:
            self.layer.append(
                nn.BatchNorm1d(N_out, affine=affine))

        # activation function
        self.layer.append(nn.ReLU(inplace=True))

        # variational dropout
        if dropout:
            self.layer.append(VariationalDropout(N_out))


    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.layer:
            x = m(x)
        return x


def stack_layers(N, layers=1, batchnorm=True, affine=True, dropout=False):
    '''
    :param N: scalar number of input and output weights.
    :param layers: scalar number of layers to stack.
    :param batchnorm: boolean for including batch normalization in each layer.
    :param affine: boolean for estimating affine weights in batch normalization.
    :param dropout: boolean for including variational dropout in each layer.
    '''
    # sequence of modules
    stack = nn.ModuleList([])
    for _ in range(layers):
        stack.append(Layer(N, N,
            batchnorm=batchnorm, affine=affine, dropout=dropout))
    return stack


class Embedding(nn.Module):
    def __init__(self, counts, params):
        '''
        :param counts: dictionary of scalar input sizes.
        :param params: dictionary of neural net parameters.
        '''
        super(Embedding, self).__init__()

        # first stack of layers: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = stack_layers(v,
                layers=params['layers_embedding'],
                batchnorm=params['batchnorm'],
                affine=params['affine'])

        # second layer: concatenation
        N = sum(counts.values())
        self.layer2 = stack_layers(N,
            layers=params['layers_embedding'],
            batchnorm=params['batchnorm'],
            affine=params['affine'])


    def forward(self, x):
        '''
        x: OrderedDict() with a superset of the keys in self.k
        '''
        # first layer
        l = []
        for k, mlist in self.layer1.items():
            x_k = x[k]
            for m in mlist:
                x_k = m(x_k)
            l.append(x_k)

        # concatenate
        x = torch.cat(l, dim=1)

        # pass through second embedding
        for m in self.layer2:
            x = m(x)

        return x


class FullyConnected(nn.Module):
    def __init__(self, N_in, N_out, params, dropout=True):
        '''
        N_in: scalar number of input weights.
        N_out: scalar number of output parameters.
        :param params: dictionary of neural net parameters.
        '''
        super(FullyConnected, self).__init__()
        # intermediate layer
        self.seq = nn.ModuleList([Layer(N_in, params['hidden'],
            batchnorm=params['batchnorm'], 
            affine=params['affine'],
            dropout=dropout)])

        # fully connected network
        self.seq += stack_layers(params['hidden'],
            layers=params['layers_full']-1,
            batchnorm=params['batchnorm'],
            affine=params['affine'],
            dropout=dropout)

        # output layer
        self.seq += [nn.Linear(params['hidden'], N_out)]


    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.seq:
            x = m(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, sizes, params, dropout=True):
        '''
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param params: dictionary of neural net parameters.
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
            d[name] = Embedding(counts, params)
            total += sum(counts.values())
        self.nn0 = nn.ModuleDict(d)

        # fully connected
        self.nn1 = FullyConnected(
            total, sizes['out'], params, dropout=dropout)


    def forward(self, x):
        '''
        x: OrderedDict()
        '''
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        return self.nn1(torch.cat(l, dim=1))
