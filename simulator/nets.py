import torch, torch.nn as nn
from torch.nn.utils import rnn
from constants import *


class FeedForward(nn.Module):
    def __init__(self, N_in, N_out, hidden):
        # super constructor
        super(FeedForward, self).__init__()

        # activation function
        f = nn.Sigmoid()

        # initial layer
        self.seq = nn.ModuleList([nn.Linear(N_in, hidden)])

        # intermediate layers
        for i in range(LAYERS-1):
            self.seq.append(nn.Linear(hidden, hidden))
            self.seq.append(f)

        # output layer
        self.seq.append(nn.Linear(hidden, N_out))


    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x


class RNN(nn.Module):
    def __init__(self, N_fixed, N_time, N_out, hidden):

        # super constructor
        super(RNN, self).__init__()

        # initial hidden nodes
        self.h0 = FeedForward(N_fixed, hidden, hidden)

        # lstm layer
        self.rnn = nn.RNN(input_size=N_time,
                            hidden_size=hidden,
                            bias=True,
                            num_layers=LAYERS,
                            dropout=DROPOUT)

        # output layer
        self.output = nn.Linear(hidden, N_out)


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time, steps):
        x_fixed = x_fixed.repeat(LAYERS, 1, 1)
        theta, _ = self.rnn(x_time, self.h0(x_fixed))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=steps)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta)