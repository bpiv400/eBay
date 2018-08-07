
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch import optim
import numpy as np
import math
import sys
import os
import pickle


class InitProcessRNN(nn.Module):
    def __init__(self, offr_size, output_size, targ_hidden_size=None, org_hidden_size=None,
                 bet_hidden_size=None, layers=1):
        # super constructor
        super(InitProcessRNN, self).__init__()
        # error checking for required argument
        if org_hidden_size is None:
            raise ValueError("Original hidden size must be provided")
        else:
            self.org_hidden_size = org_hidden_size

        # fill in default arguments
        if bet_hidden_size is None:
            self.bet_hidden_size = self.org_hidden_size
        else:
            self.bet_hidden_size = bet_hidden_size

        if targ_hidden_size is None:
            self.targ_hidden_size = self.org_hidden_size
        else:
            self.targ_hidden_size = targ_hidden_size

        # store  additional properties
        self.offr_size = offr_size
        self.output_size = output_size
        self.layers = layers

        # first input layer
        self.input_layer = nn.Linear(
            self.org_hidden_size, self.bet_hidden_size)

        # pre rnn layer
        self.prernn_layer = nn.Linear(
            self.bet_hidden_size, self.targ_hidden_size)

        # rnn layer
        self.rnn = nn.RNN(input_size=self.offr_size,
                          hidden_size=self.hidden_size,
                          num_layers=layers,
                          nonlinearity='relu',
                          bias=True,
                          dropout=0)
        # output layer
        self.oh = nn.Linear(self.hidden_size, self.output_size)

        # filler input for init_state
        self.batch_h0 = None

    def set_init_state(self, init_state):
        self.batch_h0 = init_state

    def forward(self, x):
        # x should be a packed variable length sequence of shape
        # max_seq_len, batch, offr_size
        # h_0 should be a tensor containing the initial hidden state
        # of size (1, batch, hidden_size)
        # apply  input layer to batch_h0
        self.batch_h0 = self.input_layer(self.batch_h0)
        # should transform to size 1, batch, self.bet_hidden_size
        # apply relu nonlinearity then pass through fully connected layer
        self.batch_h0 = F.relu(self.batch_h0)
        self.batch_h0 = self.prenn_layer(self.batch_h0)
        # self.rnn should output a tensor of (seq_len, batch, hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        # apply output layer
        x = self.oh(x)
        # x should now be (seq_len, batch, num_classes)
        # swap the second and last dimension to make output the shape expected
        # by cross entropy
        x = x.transpose(1, 2)
        # output result should have dimension (seq_len, num_classes, batch_size)
        return x


class SimpleRNN(nn.Module):
    def __init__(self, offr_size, output_size, hidden_size, layers=1):
        # super constructor
        super(SimpleRNN, self).__init__()

        # store properties
        self.hidden_size = hidden_size
        self.offr_size = offr_size
        self.output_size = output_size
        self.layers = layers

        # rnn layer
        self.rnn = nn.RNN(input_size=self.offr_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.layers,
                          nonlinearity='relu',
                          bias=True,
                          dropout=0)
        # output layer
        self.oh = nn.Linear(self.hidden_size, self.output_size)

        # filler input for init_state
        self.batch_h0 = None

    def set_init_state(self, init_state):
        self.batch_h0 = init_state

    def forward(self, x):
        # x should be a packed variable length sequence of shape
        # max_seq_len, batch, offr_size
        # h_0 should be a tensor containing the initial hidden state
        # of size (1, batch, hidden_size)
        if self.batch_h0 is not None:
            x, _ = self.rnn(x, self.batch_h0)
        else:
            x, _ = self.rnn(x)
        # self.rnn should output a tensor of (seq_len, batch, hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        # apply output layer
        x = self.oh(x)
        # x should now be (seq_len, batch, num_classes)
        # swap the second and last dimension to make output the shape expected
        # by cross entropy
        x = x.transpose(1, 2)
        # output result should have dimension (seq_len, num_classes, batch_size)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, offr_size, output_size, hidden_size, layers=1):
        # super constructor
        super(SimpleLSTM, self).__init__()

        # store properties
        self.hidden_size = hidden_size
        self.offr_size = offr_size
        self.output_size = output_size
        self.layers = layers

        # rnn layer
        if self.mode == 'lstm':
            self.rnn = nn.LSTM(input_size=self.offr_size,
                               hidden_size=self.hidden_size,
                               num_layers=layers,
                               bias=True,
                               dropout=0)
            else
        # output layer
        self.oh = nn.Linear(self.hidden_size, self.output_size)

        # filler input for init_state
        self.batch_h0 = None

    def set_init_state(self, init_state):
        self.batch_h0 = init_state

    def forward(self, x):
        # x should be a packed variable length sequence of shape
        # max_seq_len, batch, offr_size
        # h_0 should be a tensor containing the initial hidden state
        # of size (1, batch, hidden_size)
        if self.batch_h0 is not None:
            x, _ = self.rnn(x, self.batch_h0)
        else:
            x, _ = self.rnn(x)
        # self.rnn should output a tensor of (seq_len, batch, hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        # apply output layer
        x = self.oh(x)
        # x should now be (seq_len, batch, num_classes)
        # swap the second and last dimension to make output the shape expected
        # by cross entropy
        x = x.transpose(1, 2)
        # output result should have dimension (seq_len, num_classes, batch_size)
        return x


def get_model_class(exp_name):
    # returns the class of the model corresponding to the name of the
    # experiment
    if 'simp' or 'cat' in exp_name:
        if 'lstm' not in exp_name:
            net = SimpleRNN
        else:
            net = SimpleLSTM
    else:
        raise ValueError(
            'Experiment name does not determine whether concat or simp')
    return net
