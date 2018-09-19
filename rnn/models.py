
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch import optim
import numpy as np
import math
import sys
import torch.nn.functional as F
import os
import pickle
import re


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
            self.rnn = nn.LSTM(input_size=self.offr_size,
                               hidden_size=self.targ_hidden_size,
                               num_layers=self.layers,
                               nonlinearity='relu',
                               bias=True,
                               dropout=0)
        else:
            self.rnn = nn.RNN(input_size=self.offr_size,
                              hidden_size=self.targ_hidden_size,
                              num_layers=self.layers,
                              nonlinearity='relu',
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
        input_tens = self.prernn_layer(input_tens)
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
        # lstm handling
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

    @staticmethod
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

    @staticmethod
    def get_hidden_size(const_vals, exp_name):
        if 'hidn' in exp_name:
            layr_match = re.search('hidn', exp_name)
            # grab the index after the end of the match
            last_ind = layr_match.span(0)[1]
            # extract the corresponding substring beginning at that index
            exp_name_sub = exp_name[last_ind:]
            # match all consecutive numbers
            num_match = re.search('[0-9]*', exp_name_sub)
            # extract number and convert to int
            targ_size = int(num_match.group(0))
        else:
            targ_size = const_vals.shape[2]
        return targ_size

    @staticmethod
    def get_bet_size(exp_name):
        if 'init' in exp_name:
            if 'bet' in exp_name:
                bet_match = re.search('bet', exp_name)
                # grab the index after the end of the match
                last_ind = bet_match.span(0)[1]
                # extract the corresponding substring beginning at that index
                exp_name_sub = exp_name[last_ind:]
                # match all consecutive numbers
                num_match = re.search('[0-9]*', exp_name_sub)
                # extract number and convert to int
                targ_size = int(num_match.group(0))
            else:
                targ_size = None
        else:
            targ_size = None
        return targ_size

    @staticmethod
    def get_num_classes(midpoint_ser):
        if midpoint_ser.index.contains(-100):
            return len(midpoint_ser.index) - 1
        else:
            return len(midpoint_ser.index)

    @staticmethod
    def increase_hidden_size(const_vals, targ_size):
        # grab current number of hidden units
        curr_size = const_vals.shape[2]
        # return original if target is less than current size
        if curr_size >= targ_size:
            return const_vals
        # determine how many additional 0's must be added
        size_diff = targ_size - curr_size
        # generate tensor of necessary size
        # first and last dimensions must match const_vals
        empty_arr = np.zeros(
            (const_vals.shape[0], const_vals.shape[1], size_diff))
        # append empty values to current vals
        const_vals = np.append(const_vals, empty_arr, axis=2)
        return const_vals

    @staticmethod
    def get_num_layers(exp_name):
        if 'layr' in exp_name:
            # find where layr is in the experiment name
            layr_match = re.search('layr', exp_name)
            # grab the index after the end of the match
            last_ind = layr_match.span(0)[1]
            # extract the corresponding substring beginning at that index
            exp_name_sub = exp_name[last_ind:]
            # match all consecutive numbers
            num_match = re.search('[0-9]*', exp_name_sub)
            # extract number and convert to int
            num_layers = int(num_match.group(0))
        else:
            num_layers = 1
        return num_layers

    @staticmethod
    def get_zeros(exp_name):
        return 'zero' in exp_name

    @staticmethod
    def get_lstm(exp_name):
        return 'lstm' in exp_name

    @staticmethod
    def get_init(exp_name):
        return 'init' in exp_name

    @staticmethod
    def transform_hidden_state(*args, **kwargs):
        '''
        Args:
            const_vals: 3-dimensional ndarray with dims 1, batch_size, hidden_size
            exp_name: optional string giving the name of the experiment
        Keyword Args:
            params_dict: optional parameter dictionary including keys for
            init, zeros, num_layers, and targ_hidden_size
        Returns:
            Tuple of 3-dimensional np.ndarray with dims (num_layers, batch_size, targ_hidden_size),
            num_layers (int), targ_hidden_size (int)
        '''
        if len(args) == 0:
            raise ValueError("Must take at least 1 argument--const values")
        const_vals = args[0]
        if len(args) == 1:
            if 'params_dict' not in kwargs:
                raise ValueError(
                    "If exp_name not given, params dict must be given")
            else:
                params_dict = kwargs['params_dict']
                init = params_dict['init']
                zeros = params_dict['zeros']
                num_layers = params_dict['num_layers']
                targ_hidden_size = params_dict['targ_hidden_size']
        else:
            # parse experiment name from args
            exp_name = args[1]
            # parse experiment name for important variables
            init = SimpleRNN.get_init(exp_name)
            zeros = SimpleRNN.get_zeros(exp_name)
            num_layers = SimpleRNN.get_num_layers(exp_name)
            targ_hidden_size = SimpleRNN.get_hidden_size(const_vals, exp_name)
        # increase size of hidden state if necessary
        if not init:
            const_vals = SimpleRNN.increase_hidden_size(
                const_vals, targ_hidden_size)
        # increase the number of layers if necessary
        if num_layers > 1:
            const_vals = SimpleRNN.increase_num_layers(
                const_vals, num_layers, zeros)
        if len(args) == 2:
            return const_vals, num_layers, targ_hidden_size
        else:
            return const_vals

    @staticmethod
    def increase_num_layers(const_vals, num_layers, zeros=True):
        # this gives the total number of layers of rnns
        # adjust to give layers of starting values that must be added
        # since the constant features ( or 0's, should already occupy first layer)
        num_add = num_layers - 1
        # grab sizes of other dimensions
        batch = const_vals.shape[1]
        hidden_size = const_vals.shape[2]
        # iterate over values through num, add a layer of 0's for each
        for _ in range(num_add):
            if zeros:
                # initialize empty array to add
                empty_arr = np.zeros((1, batch, hidden_size))
                # append along the first axis
            else:
                empty_arr = np.expand_dims(const_vals[0, :, :], axis=0)
            const_vals = np.append(const_vals, empty_arr, axis=0)
        return const_vals
