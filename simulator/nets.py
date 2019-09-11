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
        self.seq = nn.ModuleList([nn.Linear(sizes['fixed'], params['ff_hidden']),
            f, nn.Dropout(p=DROPOUT)])

        # intermediate layers
        for i in range(params['ff_layers']-2):
            self.seq.append(nn.Linear(params['ff_hidden'], params['ff_hidden']))
            self.seq.append(f)
            self.seq.append(nn.Dropout(p=DROPOUT))

        # output layer
        if toRNN:
            self.seq.append(nn.Linear(params['ff_hidden'], params['rnn_hidden']))
        else:
            self.seq.append(nn.Linear(params['ff_hidden'], sizes['out']))


    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x


class RNN(nn.Module):
    def __init__(self, params, sizes):

        # super constructor
        super(RNN, self).__init__()

        # save parameters to self
        self.layers = int(params['rnn_layers'])
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(params, sizes, toRNN=True)

        # lstm layer
        self.rnn = nn.RNN(input_size=sizes['time'],
                            hidden_size=params['rnn_hidden'],
                            num_layers=self.layers,
                            dropout=DROPOUT)

        # output layer
        self.output = nn.Linear(params['rnn_hidden'], sizes['out'])


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        x_fixed = x_fixed.repeat(self.layers, 1, 1)
        theta, _ = self.rnn(x_time, self.h0(x_fixed))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta)