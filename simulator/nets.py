import torch.nn as nn
from collections import OrderedDict
from constants import *


class Layer(nn.Module):
    def __init__(self, N_in, N_out, dropout=False):
        super(Layer, self).__init__()

        # sequence of modules
        self.seq = nn.ModuleList(
            [nn.Linear(N_in, N_out), nn.BatchNorm1d(N_out), F])

        # add dropout
        if dropout:
            self.seq.append(nn.Dropout(p=DROPOUT))

    def forward(self, x):
        for m in self.seq:
            x = m(x)
        return x


class Embedding(nn.Module):
    def __init__(self, counts):
        super(Embedding, self).__init__()

        # first layer: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = Layer(v, v)

        # second layer: concatenation
        N = sum(counts.values())
        self.layer2 = Layer(N, N)

    def forward(self, x):
        """
        x: OrderedDict() with a superset of the keys in self.k
        """
        # first layer
        l = []
        for k, m in self.layer1.items():
            l.append(m(x[k]))

        # concatenate
        x = torch.cat(l, dim=1)

        # pass through second embedding
        return self.layer2(x)


class Cartesian(nn.Module):
    def __init__(self, counts):
        '''
        counts: OrderedDict of input sizes
        f: activation function (e.g., nn.ReLU())
        '''
        super(Cartesian, self).__init__()

        # create all pairwise comparisons
        keys = list(counts.keys())
        d, k = OrderedDict(), len(keys)
        for i in range(k-1):
            for j in range(i+1,k):
                N = counts[keys[i]] + counts[keys[j]]
                newkey = '-'.join([keys[i], keys[j]])
                d[newkey] = Layer(N, N)
        self.pairs = nn.ModuleDict(d)

    def forward(self, x):
        '''
        x: dictionary with outputs from separate Embeddings
        '''
        l = []
        for key, m in self.pairs.items():
            key1, key2 = key.split('-')
            z = torch.cat((x[key1], x[key2]), axis=1)
            l.append(m(z))

        return torch.cat(l, axis=1)


class FullyConnected(nn.Module):
    def __init__(self, N_in, N_out):
        super(FullyConnected, self).__init__()

        # intermediate layer
        self.seq = nn.ModuleList([Layer(N_in, HIDDEN, dropout=True)])

        # fully connected network
        for i in range(LAYERS-1):
            self.seq.append(
                Layer(HIDDEN, HIDDEN, dropout=(i < LAYERS-2)))

        # output layer
        self.seq.append(nn.Linear(HIDDEN, N_out))

    def forward(self, x):
        for m in self.seq:
            x = m(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, sizes, toRNN=False):
        super(FeedForward, self).__init__()

        # embedding
        d = OrderedDict()
        for name, group in EMBEDDING_GROUPS.items():
            counts = {k: v for k, v in sizes['x'].items() if k in group}
            d[name] = Embedding(counts)
        self.nn0 = nn.ModuleDict(d)

        # cartesian
        counts = OrderedDict()
        for name, group in EMBEDDING_GROUPS.items():
            counts[name] = sum(
                [v for k, v in sizes['x'].items() if k in group])
        self.nn1 = Cartesian(counts)

        # fully connected
        N_in = sum(counts.values()) * (len(counts) - 1)
        N_out = HIDDEN if toRNN else sizes['out']
        self.nn2 = FullyConnected(N_in, N_out)

    def forward(self, x):
        # separate embeddings
        d = OrderedDict()
        for name, group in EMBEDDING_GROUPS.items():
            d[name] = self.nn0[name](x)
        
        # cartesian, then fully connected
        return self.nn2(self.nn1(d))


class LSTM(nn.Module):
    def __init__(self, sizes):
        super(LSTM, self).__init__()

        # save parameters to self
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(sizes, toRNN=True)
        self.c0 = FeedForward(sizes, toRNN=True)

        # rnn layer
        self.rnn = nn.LSTM(input_size=sizes['x_time'],
                           hidden_size=HIDDEN,
                           batch_first=True)

        # output layer
        self.output = nn.Linear(HIDDEN, sizes['out'])

    # output discrete weights and parameters of continuous components
    def forward(self, x, x_time):
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
