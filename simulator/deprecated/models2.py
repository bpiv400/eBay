import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils import *

MAX_TURNS = 3
TOL_HALF = 0.02  # count concessions within this range as 1/2


class Simulator(nn.Module):
    def __init__(self, N_fixed, N_offer, N_hidden, N_out, N_layers, dropout, lstm):
        # super constructor
        super(Simulator, self).__init__()

        # variables
        self.N_layers = N_layers

        # initial hidden nodes
        self.h0 = nn.Linear(N_fixed, N_hidden)

        # activation function
        self.f = nn.Sigmoid()

        # rnn / lstm layer
        self.lstm = lstm
        if lstm:
            self.c0 = nn.Linear(N_fixed, N_hidden)  # initial cell nodes
            self.rnn = nn.LSTM(input_size=N_offer, hidden_size=N_hidden,
                               bias=True, num_layers=N_layers, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size=N_offer, hidden_size=N_hidden,
                              bias=True, num_layers=N_layers, dropout=dropout)

        # output layer
        self.output = nn.Linear(N_hidden, N_out)

        # number of discrete masses
        self.k = max(1, N_out - 2)
        self.N_out = N_out

    def forward(self, x_fixed, x_offer):
        # initialize model
        x_fixed = x_fixed.repeat(self.N_layers, 1, 1)
        init = self.f(self.h0(x_fixed))
        if self.lstm:
            init = (init, self.f(self.c0(x_fixed)))
        theta, _ = self.rnn(x_offer, init)

        # ensure that predictions are padded to MAX_TURNS
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=MAX_TURNS)

        # output layer w/exponential transform: (seq_len, batch_size, N_output)
        return torch.exp(self.output(theta))

    def loss_forward(self, theta, y):
        """
        Computes the nll given delay parameters and targets y
        """
        # parameters of mixture model
        g = theta[:, :, 0] * 1.0
        a = theta[:, :, 1] + 1
        b = theta[:, :, 2] + 1
        # treat slices as leaves
        g.retain_grad()
        a.retain_grad()
        b.retain_grad()
        # probability of full delay
        p = torch.div(g, 1 + g)

        # indices
        idx = {}
        idx[2] = y == 1  # indices for full delay
        idx[3] = ~idx[2] * ~torch.isnan(y)  # indices for small delay

        dist = torch.distributions.beta.Beta(a, b)  # initialize distribution
        ln_density = dist.log_prob(y)[idx[3]] * 1.0
        lnp_early = torch.log(p[idx[3]] * 1.0)
        lnp_late = torch.log(p[idx[2]] * 1.0)
        print(torch.isnan(ln_density).any())
        print(torch.isnan(lnp_early).any())
        print(torch.isnan(lnp_late).any())
        # treat slices as leaves
        ln_density.retain_grad()
        lnp_early.retain_grad()
        lnp_late.retain_grad()
        ll = torch.sum(ln_density) + torch.sum(lnp_early) + torch.sum(lnp_late)
        ll.retain_grad()
        print(ll)
        #  final leaf check
        return -ll


class ConLoss(Function):
    '''
    Computes the negative log-likelihood for the slr concessions model.

    Inputs to forward:
        - theta (N_turns, mbsize, 5):
            [gamma_rej, gamma_acc, gamma_50, alpha, beta]
        - y (N_turns, mbsize): slr concessions

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, theta, y):
        # parameters of mixture model
        g = theta[:, :, 0:3]
        a = theta[:, :, 3] + 1
        b = theta[:, :, 4] + 1
        p = torch.div(g, 1 + torch.sum(g, 2, keepdim=True))

        # indices
        idx = {}
        idx[0] = y == 0
        idx[1] = y == 1
        idx[2] = torch.abs(y - 0.5) < TOL_HALF
        idx[3] = ~idx[0] * ~idx[1] * ~idx[2] * ~torch.isnan(y)

        # store variables for backward method
        ctx.g = g
        ctx.a = a
        ctx.b = b
        ctx.y = y
        ctx.idx = idx

        # log-likelihood
        lnp = ln_beta_pdf(a, b, y)
        ll = torch.sum(torch.log(1 - torch.sum(p, 2)[idx[3]]) + lnp[idx[3]])
        for i in range(3):
            ll += torch.sum(torch.log(p[idx[i], [i]]))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        # initialize floats
        idx = {key: val.float() for key, val in ctx.idx.items()}
        dg = torch.zeros(ctx.g.size())

        # convert to double for gradcheck
        if ctx.a.type() == 'torch.DoubleTensor':
            idx = {key: val.double() for key, val in ctx.idx.items()}
            dg = dg.double()

        # gradients for beta parameters
        da = torch.unsqueeze(dlnLda(ctx.a, ctx.b, ctx.y, idx[3]), dim=2)
        db = torch.unsqueeze(dlnLda(ctx.b, ctx.a, 1-ctx.y, idx[3]), dim=2)

        # gradients for discrete probabilities
        dg3 = torch.pow(1+torch.sum(ctx.g, 2), -1)
        for i in range(3):
            dg[:, :, i] = idx[i] * torch.pow(ctx.g[:, :, i], -1) - dg3

        return grad_output * -torch.cat((dg, da, db), dim=2), None


class MsgLoss(Function):
    '''
    Computes the negative log-likelihood for the slr msg model.

    Inputs to forward:
        - theta (N_turns, mbsize): [gamma_msg]
        - y (N_turns, mbsize): slr days

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, theta, y):
        # parameters of mixture model
        g = theta.squeeze()
        p = torch.div(g, 1 + g)

        # indices
        idx = {}
        idx[1] = y == 1
        idx[0] = ~idx[1] * ~torch.isnan(y)

        # store variables for backward method
        ctx.g = g
        ctx.y = y
        ctx.idx = idx

        # log-likelihood
        ll = torch.sum(torch.log(1 - p[idx[0]]))
        ll += torch.sum(torch.log(p[idx[1]]))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        # initialize floats
        idx = {key: val.float() for key, val in ctx.idx.items()}
        dg = torch.zeros(ctx.g.size())

        # convert to double for gradcheck
        if ctx.g.type() == 'torch.DoubleTensor':
            idx = {key: val.double() for key, val in ctx.idx.items()}
            dg = dg.double()

        # gradient for discrete probabilities
        dg = idx[1] * torch.pow(ctx.g, -1) - torch.pow(1 + ctx.g, -1)
        dg = torch.unsqueeze(dg, dim=2)

        return grad_output * -dg, None
