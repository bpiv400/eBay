import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, 
                 dropout_method='pytorch', layernorm=False, affine=True):
        super(LSTM, self).__init__()

        # save parameters to self
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        
        # linear transformations
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        # initialize parameters
        self.reset_parameters()

        # dropout method
        assert(dropout_method.lower() in ['pytorch', 'gal', 'semeniuta'])
        self.dropout_method = dropout_method

        # layer normalization (for the hidden state)
        if layernorm:
            self.ln_cell = nn.LayerNorm(hidden_size, 
                elementwise_affine=affine)
        else:
            self.ln_cell = nn.Identity()


    def sample_mask(self):
        '''
        Call before every minibatch during training when using Gal dropout.
        '''
        keep = 1.0 - self.dropout
        self.mask = V(torch.bernoulli(T(1, self.hidden_size).fill_(keep)))


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = self.ln_cell(torch.mul(c, f_t) + torch.mul(i_t, g_t))
        h_t = torch.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(torch.mul(h_t, self.mask).data)
                h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)
