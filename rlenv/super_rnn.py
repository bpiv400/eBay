
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


class SimpleRNN(nn.module):
    def __init__(self, n_fixed, n_hidden, n_seq_feats, n_hidden_layers):
        super(SimpleRNN, self).__init__()

        # initial hidden nodes
        self.h0 = nn.Linear(n_fixed, n_hidden)

        # initial cell nodes
        self.c0 = nn.Linear(n_fixed, n_hidden)

        # activation function
        self.f = nn.Sigmoid()

        # lstm layer
        self.lstm = nn.LSTM(input_size=n_seq_feats, hidden_size=n_hidden,
                            bias=True, num_layers=n_hidden_layers, nonlinearity="relu")

    # def forward(self, x):
    #     pass
