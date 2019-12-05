import torch.nn as nn
from constants import *


class Embedding(nn.Module):
    def __init__(self, sizes):
        # super constructor
        super(Embedding, self).__init__()

        # save dictionary of input feat sizes
        self.k = sizes['x']

        # activation function
        f = nn.ReLU()

        # embeddings layer(s)
        d = {}
        for k, v in self.k.items():
            l = []
            for i in range(2):
                l += [nn.Linear(v, v), nn.BatchNorm1d(v), f]
            d[k] = nn.ModuleList(l)
        self.embedding = nn.ModuleDict(d)

    def forward(self, x):
        """
        x: OrderedDict() with same keys as self.k
        """
        # ensure that keys of x and self.k are of the same length
        assert len(x.keys()) == len(self.k.keys())

        # separate embeddings for input feature components
        l = []
        for key in self.k.keys():
            x_k = x[key]
            for m in self.embedding[key]:
                x_k = m(x_k)
            l.append(x_k)

        # concatenate and return
        return torch.cat(l, dim=1)


class FullyConnected(nn.Module):
    def __init__(self, sizes, toRNN=False):
        # super constructor
        super(FullyConnected, self).__init__()

        # activation function
        f = nn.ReLU()

        # intermediate layer
        self.seq = nn.ModuleList(
            [nn.Linear(sum(sizes['x'].values()), HIDDEN), 
             nn.BatchNorm1d(HIDDEN), f])

        # fully connected network
        for i in range(LAYERS-1):
            self.seq.append(nn.Dropout(p=DROPOUT))
            self.seq.append(nn.Linear(HIDDEN, HIDDEN))
            self.seq.append(nn.BatchNorm1d(HIDDEN))
            self.seq.append(f)

        # output layer
        if toRNN:
            self.seq.append(nn.Linear(HIDDEN, HIDDEN))
        else:
            self.seq.append(nn.Linear(HIDDEN, sizes['out']))

    def forward(self, x):
        """
        x: OrderedDict() with same keys as self.k
        """
        # separate embeddings for input feature components
        for m in self.seq:
            x = m(x)
        return x


class LSTM(nn.Module):
    def __init__(self, sizes):

        # super constructor
        super(LSTM, self).__init__()

        # save parameters to self
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FullyConnected(sizes, toRNN=True)
        self.c0 = FullyConnected(sizes, toRNN=True)

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
