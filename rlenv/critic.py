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


class Critic(nn.Module):
    MAX_SEQ_LENGTH = 4

    def __init__(self, n_fixed, n_hidden, n_seq_feats, n_hidden_layers):
        # initialize common layers
        super.__init__(n_fixed, n_hidden, n_seq_feats, n_hidden_layers)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x, consts):
        init_hidden = self.h0(consts)
        init_hidden = self.f(init_hidden)
        init_cell = self.c0(consts)
        init_cell = self.f(init_cell)

        x = self.lstm(x, (init_hidden, init_cell))
        # ensure that predictions are padded to MAX_SEQ_LEN
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, total_length=self.MAX_SEQ_LEN)

        # output layer
        x = self.output(x)

        x = torch.clamp(x, 0, 1)

        return x
