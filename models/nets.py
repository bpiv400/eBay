import torch, torch.nn as nn
from collections import OrderedDict
from constants import *


class VariationalDropout(nn.Module):
    def __init__(self, N):
        super(VariationalDropout, self).__init__()
        
        self.log_alpha = nn.Parameter(torch.Tensor(N))
        self.log_alpha.data.fill_(-3)
        
    def kl_reg(self):
        # calculate KL divergence
        kl = 0.63576 * (torch.sigmoid(1.8732 + 1.48695 * self.log_alpha) - 1) \
            - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -torch.sum(kl)
    
    def forward(self, x):
        # additive N(0, alpha * x^2) noise
        if self.training:
            sigma = torch.sqrt(torch.exp(self.log_alpha)) * torch.abs(x)
            eps = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            return x + sigma * eps
        else:
            return x


class Layer(nn.Module):
    def __init__(self, N_in, N_out, dropout=True):
        '''
        N_in: scalar number of input weights
        N_out: scalar number of output weights
        '''
        super(Layer, self).__init__()

        # one layer is a weight matrix, batch normalization and activation
        if dropout:
            self.layer = nn.Sequential(nn.Linear(N_in, N_out), 
                nn.BatchNorm1d(N_out, affine=False), F, VariationalDropout(N_out))
        else:
            self.layer = nn.Sequential(nn.Linear(N_in, N_out), 
                nn.BatchNorm1d(N_out, affine=False), F)

    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.layer:
            x = m(x)
        return x


class Stack(nn.Module):
    def __init__(self, N, layers=1, dropout=True):
        '''
        N: scalar number of input and output weights
        layers: scalar number of layers to stack
        '''
        super(Stack, self).__init__()

        # sequence of modules
        self.stack = nn.ModuleList(
            [Layer(N, N, dropout=dropout) for _ in range(layers)])
        
    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.stack:
            x = m(x)
        return x


class Embedding(nn.Module):
    def __init__(self, counts):
        '''
        sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        '''
        super(Embedding, self).__init__()

        # first stack of layers: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = Stack(v, layers=LAYERS_EMBEDDING, dropout=False)

        # second layer: concatenation
        N = sum(counts.values())
        self.layer2 = Stack(N, layers=LAYERS_EMBEDDING, dropout=False)

    def forward(self, x):
        '''
        x: OrderedDict() with a superset of the keys in self.k
        '''
        # first layer
        l = []
        for k, m in self.layer1.items():
            l.append(m(x[k]))

        # concatenate
        x = torch.cat(l, dim=1)

        # pass through second embedding
        return self.layer2(x)


class FullyConnected(nn.Module):
    def __init__(self, N_in, N_out, dropout):
        '''
        sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        N_in: scalar number of input weights
        N_out: scalar number of output parameters
        '''
        super(FullyConnected, self).__init__()

        # intermediate layer
        self.seq = nn.ModuleList([Layer(N_in, HIDDEN, dropout=dropout)])

        # fully connected network
        self.seq.append(Stack(HIDDEN, layers=LAYERS_FULL-2, dropout=dropout))

        # last layer has no dropout
        self.seq.append(Stack(HIDDEN, dropout=False))

        # output layer
        self.seq.append(nn.Linear(HIDDEN, N_out))

    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.seq:
            x = m(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, sizes, dropout=True, toRNN=False):
        '''
        sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        dropout: True if using dropout for fully-connected layers
        toRNN: True if FeedForward initialized hidden state of recurrent network
        '''
        super(FeedForward, self).__init__()

        # expand embeddings for offer models
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
        N_in = sum(counts.values())
        self.nn1 = FullyConnected(total, 
            HIDDEN if toRNN else sizes['out'], dropout)

    def forward(self, x):
        '''
        x: OrderedDict()
        '''
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        return self.nn1(torch.cat(l, dim=1))


class LSTM(nn.Module):
    def __init__(self, sizes, dropout=True):
        '''
        sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        dropout: True if using dropout for fully-connected layers
        '''
        super(LSTM, self).__init__()

        # save parameters to self
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(sizes, dropout, toRNN=True)
        self.c0 = FeedForward(sizes, dropout, toRNN=True)

        # rnn layer
        self.rnn = nn.LSTM(input_size=sizes['x_time'],
                           hidden_size=HIDDEN,
                           batch_first=True)

        # output layer
        self.output = nn.Linear(HIDDEN, sizes['out'])

    def forward(self, x, x_time):
        '''
        x: OrderedDict()
        x_time: tensor of shape [mbsize, sizes['steps'], sizes['x_time']]
        '''
        # initialize hidden state
        hidden = (self.h0(x).unsqueeze(dim=0), self.c0(x).unsqueeze(dim=0))

        # update hidden state recurrently
        theta, _ = self.rnn(x_time, hidden)

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: (batch_size, seq_len, N_output)
        return self.output(theta).squeeze()

    def init(self, x=None):
        hidden = (self.h0(x).unsqueeze(dim=0), self.c0(x).unsqueeze(dim=0))
        return hidden

    def step(self, x_time=None, hidden=None, x=None):
        """
        Executes 1 step of the recurrent network

        :param x_time:
        :param x:
        :param hidden:
        :return:
        """
        if hidden is None:
            hidden = self.init(x=x)
        theta, hidden = self.rnn(x_time.unsqueeze(0), hidden)
        # output layer: (seq_len, batch_size, N_out)
        return self.output(theta), hidden
