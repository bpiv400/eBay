import torch, torch.nn as nn
from torch.nn.utils import rnn
from constants import *

MAX = 1.0


class FeedForward(nn.Module):
    def __init__(self, params, sizes, toRNN=False):
        # super constructor
        super(FeedForward, self).__init__()

        # activation function
        f = nn.Sigmoid()
        #f = nn.Hardtanh(max_val=MAX)

        # initial layer
        self.seq = nn.ModuleList(
            [nn.Linear(sizes['fixed'], params['hidden']), f])

        # intermediate layers
        for i in range(params['layers']-1):
            self.seq.append(nn.Dropout(p=DROPOUT))
            self.seq.append(
                nn.Linear(params['hidden'], params['hidden']))
            self.seq.append(f)

        # output layer
        if toRNN:
            self.seq.append(
                nn.Linear(params['hidden'], params['hidden']))
        else:
            self.seq.append(
                nn.Linear(params['hidden'], sizes['out']))


    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x


    def simulate(self, x):
        return self.forward(x).squeeze()


class RNN(nn.Module):
    def __init__(self, params, sizes):

        # super constructor
        super(RNN, self).__init__()

        # save parameters to self
        self.layers = int(params['layers'])
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(params, sizes, toRNN=True)

        # rnn layer
        self.rnn = nn.RNN(input_size=sizes['time'],
                            hidden_size=int(params['hidden']),
                            num_layers=self.layers,
                            batch_first=True,
                            dropout=DROPOUT)

        # output layer
        self.output = nn.Linear(params['hidden'], sizes['out'])

        # for last byr turn
        self.output4 = nn.Linear(params['hidden'], 1)


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        x_fixed = x_fixed.unsqueeze(dim=0).repeat(self.layers, 1, 1)
        theta, _ = self.rnn(x_time, self.h0(x_fixed))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: split if turns == 4 (i.e., con_byr model)
        if self.steps == 4:
            return self.output(theta[:,:self.steps-1,:]), \
                self.output4(theta[:,self.steps-1,:]).squeeze()
        else:
            return self.output(theta)   # (batch_size, seq_len, N_out)


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
                           hidden_size=int(params['hidden']),
                           batch_first=True)

        # output layer
        self.output = nn.Linear(params['hidden'], sizes['out'])


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        x_fixed = x_fixed.unsqueeze(dim=0)
        theta, _ = self.rnn(x_time, (self.h0(x_fixed), self.c0(x_fixed)))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: (batch_size, seq_len, N_output)
        return self.output(theta).squeeze()


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

        # output layer: (seq_len, batch_size, N_out)
        return self.output(theta), hidden
