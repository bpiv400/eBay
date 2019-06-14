import torch, torch.nn as nn
from torch.nn.utils import rnn


class FeedForward(nn.Module):
    def __init__(self, N_in, N_out, params):
        # super constructor
        super(FeedForward, self).__init__()

        # divide dropout by 10
        dropout = params.dropout / 10

        # activation function
        f = nn.Sigmoid()

        # initial layer
        self.seq = nn.ModuleList(
            [nn.Linear(N_in, params.hidden), f, nn.Dropout(dropout)])

        # intermediate layers
        for i in range(params.ff_layers-1):
            self.seq.append(nn.Linear(params.hidden, params.hidden))
            self.seq.append(f)
            if i < params.ff_layers-2:
                self.seq.append(nn.Dropout(dropout))

        # output layer
        self.seq.append(nn.Linear(params.hidden, N_out))


    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, N_fixed, N_time, N_out, params):

        # super constructor
        super(LSTM, self).__init__()

        # save parameters to self
        self.layers = int(params.lstm_layers)

        # initial hidden nodes and LSTM cell
        self.h0 = FeedForward(N_fixed, params.hidden, params)
        self.c0 = FeedForward(N_fixed, params.hidden, params)

        # lstm layer
        self.lstm = nn.LSTM(input_size=N_time,
                            hidden_size=params.hidden,
                            bias=True,
                            num_layers=self.layers,
                            dropout=params.dropout / 10)

        # output layer
        self.output = nn.Linear(params.hidden, N_out)


    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time, steps):
        # initialize model
        x_fixed = x_fixed.repeat(self.layers, 1, 1)
        init = (self.h0(x_fixed), self.c0(x_fixed))
        theta, _ = self.lstm(x_time, init)

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=steps)

        # output layer: (seq_len, batch_size, N_output)
        return self.output(theta).squeeze()