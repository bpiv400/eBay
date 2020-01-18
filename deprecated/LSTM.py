import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from datetime import datetime as dt


class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0,
                 bidirectional=0,
                 batch_first=False,
                 layernorm=False,
                 affine=True):
        super(LSTM, self).__init__()

        # save parameters to self
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = bidirectional + 1
        self.batch_first = batch_first

        # stack modules
        layers = []
        for i in range(num_layers):
            for j in range(self.direction):
                layer = LSTMcell(input_size*self.direction,
                                 hidden_size,
                                 dropout=dropout,
                                 layernorm=layernorm,
                                 affine=affine)
                layers.append(layer)
            input_size = hidden_size
        self.layers = layers
        self.params = nn.ModuleList(layers)


    def sample_mask(self, device):
        '''
        Call before each minibatch when using Gal dropout.
        :param device: 'cuda' for GPU.
        '''
        for l in self.layers:
            l.sample_mask(device)


    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()


    def layer_forward(self, l, xs, h, reverse=False):
        '''
        return:
            xs: (seq_len, batch, hidden)
            h: (1, batch, hidden)
        '''
        if self.batch_first:
            xs = xs.permute(1, 0, 2).contiguous()
        ys = []
        for i in range(xs.size(0)):
            if reverse:
                x = xs.narrow(0, (xs.size(0)-1)-i, 1)
            else:
                x = xs.narrow(0, i, 1)
            y, h = l(x, h)
            ys.append(y)
        y = torch.cat(ys, 0)
        return y, h


    def forward(self, x, hiddens):
        if self.direction > 1:
            x = torch.cat((x, x), 2)
        if type(hiddens) != list:
            # when the hidden feed is (direction * num_layer, batch, hidden)
            tmp = []
            for idx in range(hiddens[0].size(0)):
                tmp.append((hiddens[0].narrow(0, idx, 1),
                           (hiddens[1].narrow(0, idx, 1))))
            hiddens = tmp

        new_hs = []
        new_cs = []
        for l_idx in range(0, len(self.layers), self.direction):
            l, h = self.layers[l_idx], hiddens[l_idx]
            f_x, f_h = self.layer_forward(l, x, h)
            if self.direction > 1:
                l, h  = self.layers[l_idx+1], hiddens[l_idx+1]
                r_x, r_h = self.layer_forward(l, x, h, reverse=True)

                x = torch.cat((f_x, r_x), 2)
                h = torch.cat((f_h[0], r_h[0]), 0)
                c = torch.cat((f_h[1], r_h[1]), 0)
            else:
                x = f_x
                h, c = f_h
            new_hs.append(h)
            new_cs.append(c)

        h = torch.cat(new_hs, 0)
        c = torch.cat(new_cs, 0)
        if self.batch_first:
            x = x.permute(1, 0, 2)

        return x, (h, c)


class LSTMcell(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 bias=True, 
                 dropout=0.0, 
                 dropout_method='gal', 
                 layernorm=False, 
                 affine=True):
        super(LSTMcell, self).__init__()

        # save parameters to self
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.dropout_method = dropout_method

        # linear transformations
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        # initialize weights
        self.reset_parameters()

        # error checking
        assert(dropout_method.lower() in ['pytorch', 'gal', 'semeniuta'])
        
        # layer normalization using built-in PyTorch function
        if layernorm:
            self.ln_cell = nn.LayerNorm(hidden_size, 
                elementwise_affine=affine)
        else:
            self.ln_cell = nn.Identity()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def sample_mask(self, device):
        '''
        Draws new binary mask for LSTM layer.
        :param device: 'cuda' for GPU.
        '''
        keep = 1.0 - self.dropout
        pkeep = torch.empty(1, self.hidden_size, device=device).fill_(keep)
        self.mask = torch.bernoulli(pkeep)


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
                h_t = torch.mul(h_t, self.mask) / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)

        return h_t, (h_t, c_t)
