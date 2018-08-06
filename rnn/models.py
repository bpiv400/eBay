
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
    def __init__(self, offr_size, output_size, hidden_size):
        self.hidden_size = hidden_size
        self.offr_size = offr_size
        self.output_size = output_size

        # rnn layer
        self.rnn = nn.RNN(self.offr_size, self.hidden_size,
                          1, 'relu', bias=True, dropout=0)
        # output layer
        self.oh = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h_0):
        # x should be a packed variable length sequence of shape
        # max_seq_len, batch, offr_size
        # h_0 should be a tensor containing the initial hidden state
        # of size (1, batch, hidden_size)
        x, _ = self.rnn(input=x, h_0=h_0)
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
    if 'simp' in exp_name:
        if 'lstm' not in exp_name:
            net = SimpleRNN
    elif 'cat' in exp_name:
        concat = True
        sep = False
    elif 'sep' in exp_name:
        sep = True
        concat = False
    else:
        raise ValueError(
            'Experiment name does not determine whether concat or simp')
    return net
