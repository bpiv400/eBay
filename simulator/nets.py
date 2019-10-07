import torch, torch.nn as nn
from torch.nn.utils import rnn
from constants import *


class FeedForward(nn.Module):
    def __init__(self, params, sizes, toRNN=False):
        # super constructor
        super(FeedForward, self).__init__()

        # activation function
        f = nn.Sigmoid()

        # initial layer
        self.seq = nn.ModuleList(
            [nn.Linear(sizes['fixed'], params['ff_hidden']), f])

        # intermediate layers
        for i in range(params['ff_layers']-1):
            self.seq.append(nn.Dropout(p=DROPOUT))
            self.seq.append(
                nn.Linear(params['ff_hidden'], params['ff_hidden']))
            self.seq.append(f)

        # output layer
        if toRNN:
            self.seq.append(
                nn.Linear(params['ff_hidden'], params['rnn_hidden']))
        else:
            self.seq.append(
                nn.Linear(params['ff_hidden'], sizes['out']))

    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x

    def simulate(self, x):
        return self.forward(x)


class RNN(nn.Module):
    def __init__(self, params, sizes):

        # super constructor
        super(RNN, self).__init__()

        # save parameters to self
        self.layers = int(params['rnn_layers'])
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(params, sizes, toRNN=True)

        # rnn layer
        self.rnn = nn.RNN(input_size=sizes['time'],
                            hidden_size=int(params['rnn_hidden']),
                            num_layers=self.layers,
                            batch_first=True,
                            dropout=DROPOUT)

        # output layer
        self.output = nn.Linear(params['rnn_hidden'], sizes['out'])


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        x_fixed = x_fixed.unsqueeze(dim=0).repeat(self.layers, 1, 1)
        theta, _ = self.rnn(x_time, self.h0(x_fixed))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta)


class LSTM(nn.Module):
    def __init__(self, params, sizes):

        # super constructor
        super(LSTM, self).__init__()

        # save parameters to self
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(params, sizes, toRNN=True)
        self.c0 = FeedForward(params, sizes, toRNN=True)

        # rnn layer
        self.rnn = nn.LSTM(input_size=sizes['time'],
                           hidden_size=int(params['rnn_hidden']),
                           batch_first=True)

        # output layer
        self.output = nn.Linear(params['rnn_hidden'], sizes['out'])


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        theta, _ = self.rnn(x_time, (self.h0(x_fixed), self.c0(x_fixed)))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta)

    def simulate(self, x_time, x_fixed=None, hidden=None):
        """

        :param x_time:
        :param x_fixed:
        :param hidden:
        :return:
        """
        if hidden is None:
            theta, hidden = self.rnn(x_time, (self.h0(x_fixed), self.c0(x_fixed)))
        else:
            theta, hidden = self.rnn(x_time, hidden)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta), hidden
