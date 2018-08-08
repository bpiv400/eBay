
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


class SimpleRNN(nn.Module):
    def __init__(self, offr_size, output_size, lstm=False, targ_hidden_size=None, org_hidden_size=None,
                 bet_hidden_size=None, layers=1, init_processing=False):
        # super constructor
        super(SimpleRNN, self).__init__()
        # error checking for required argument
        if org_hidden_size is None:
            raise ValueError("Original hidden size must be provided")
        else:
            self.org_hidden_size = org_hidden_size

        # determine whether initial processing is activated
        self.init_processing = init_processing

        # store  additional properties
        self.offr_size = offr_size
        self.output_size = output_size
        self.layers = layers
        self.lstm = lstm

        # initialize init_processing arguments if necessary
        if self.init_processing:
            # fill in default arguments
            if bet_hidden_size is None:
                self.bet_hidden_size = self.org_hidden_size
            else:
                self.bet_hidden_size = bet_hidden_size

            if targ_hidden_size is None:
                self.targ_hidden_size = self.org_hidden_size
            else:
                self.targ_hidden_size = targ_hidden_size

            # further initialize init_processing features if necessary
            # first input layer
            self.input_layer = nn.Linear(
                self.org_hidden_size, self.bet_hidden_size)

            # pre rnn layer
            self.prernn_layer = nn.Linear(
                self.bet_hidden_size, self.targ_hidden_size)
        else:
            self.targ_hidden_size = self.org_hidden_size

        # rnn layer
        if self.lstm:
            self.rnn = nn.RNN(input_size=self.offr_size,
                              hidden_size=self.targ_hidden_size,
                              num_layers=self.layers,
                              nonlinearity='relu',
                              bias=True,
                              dropout=0)
        else:
            self.rnn = nn.LSTM(input_size=self.offr_size,
                               hidden_size=self.targ_hidden_size,
                               num_layers=self.layers,
                               bias=True,
                               dropout=0)
        # output layer
        self.oh = nn.Linear(self.targ_hidden_size, self.output_size)

        # filler input for init_state
        self.batch_h0 = None
        self.batch_c0 = None

    def set_init_state(self, init_state):
        self.batch_h0 = init_state
        self.batch_c0 = init_state

    def __init_process(self, input_tens):
        input_tens = self.input_layer(input_tens)
        # should transform to size 1, batch, self.bet_hidden_size
        # apply relu nonlinearity then pass through fully connected layer
        input_tens = F.relu(input_tens)
        # apply transformation to hidden state
        input_tens = self.prenn_layer(input_tens)
        return input_tens

    def forward(self, x):
        # x should be a packed variable length sequence of shape
        # max_seq_len, batch, offr_size
        # h_0 should be a tensor containing the initial hidden state
        # of size (1, batch, hidden_size)
        # apply  input layer to batch_h0
        if self.init_processing:
            # self.rnn should output a tensor of (seq_len, batch, hidden_size)
            self.batch_h0 = self.__init_process(self.batch_h0)
            if self.lstm:
                self.batch_c0 = self.__init_process(self.batch_c0)
        if self.lstm:
            x, _ = self.rnn(x, (self.batch_h0, self.batch_c0))
        else:
            x, _ = self.rnn(x, self.batch_h0)

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
        #     if 'lstm' not in exp_name:
        #         net = SimpleRNN
        #     else:
        #         net = SimpleLSTM
        net = SimpleRNN
    else:
        raise ValueError(
            'Experiment name does not determine whether concat or simp')
    return net
