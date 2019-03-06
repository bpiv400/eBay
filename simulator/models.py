import torch, torch.nn as nn
from torch.autograd import Function
import numpy as np

N_OUTPUT = 5
MAX_TURNS = 3
TOL_HALF = 0.01 # count concessions within this range as 1/2

class Simulator(nn.Module):
    def __init__(self, N_fixed, N_offer, N_hidden, N_layers, dropout, is_lstm):
        # super constructor
        super(Simulator, self).__init__()

        # variables
        self.N_layers = N_layers

        # initial hidden nodes
        self.h0 = nn.Linear(N_fixed, N_hidden)

        # activation function
        self.f = nn.Sigmoid()

        # rnn / lstm layer
        self.is_lstm = is_lstm
        if is_lstm:
            self.c0 = nn.Linear(N_fixed, N_hidden)  # initial cell nodes
            self.rnn = nn.LSTM(input_size=N_offer, hidden_size=N_hidden,
                bias=True, num_layers=N_layers, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size=N_offer, hidden_size=N_hidden,
                bias=True, num_layers=N_layers, dropout=dropout)

        # output layer
        self.output = nn.Linear(N_hidden, N_OUTPUT)


    def forward(self, x_fixed, x_offer):
        # initialize model
        x_fixed = x_fixed.repeat(self.N_layers, 1, 1)
        init = self.f(self.h0(x_fixed))
        if self.is_lstm:
            init = (init, self.f(self.c0(x_fixed)))
        x, _ = self.rnn(x_offer, init)

        # ensure that predictions are padded to MAX_TURNS
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=MAX_TURNS)

        # output layer w/exponential transform: (seq_len, batch_size, N_output)
        x = torch.exp(self.output(x))

        # discrete probabilities
        p = torch.div(x[:,:,0:3], 1 + torch.sum(x[:,:,0:3], 2, keepdim=True))

        # parameters of beta distribution
        a = x[:,:,3] + 1
        b = x[:,:,4] + 1

        return p, a, b


class BetaMixtureLoss(Function):
    '''
    Computes the negative log-likelihood for the beta mixture model.

    Inputs to forward:
        - p (N_turns, mbsize, 3): p_reject (index 0), p_accept (index 1)
        and p_50 (index 2)
        - a (N_turns, mbsize): alpha parameter of beta distribution
        - b (N_turns, mbsize): beta parameter of beta distribution
        - z (N_turns, mbsize, 1): seller concessions

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, p, a, b, z):
        # indices
        z = z.squeeze(dim=2)
        idx = {}
        idx[0] = z == 0
        idx[1] = z == 1
        idx[2] = torch.abs(z - 0.5) < TOL_HALF
        idx[3] = ~idx[0] * ~idx[1] * ~idx[2] * ~torch.isnan(z)

        # store variables for backward method
        ctx.p = p
        ctx.a = a
        ctx.b = b
        ctx.z = z
        ctx.idx = idx

        # beta distribution
        lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        la = torch.mul(a-1, torch.log(z))
        lb = torch.mul(b-1, torch.log(1-z))
        lnp = la + lb - lbeta

        # log-likelihood
        ll = torch.sum(torch.log(1 - torch.sum(p, 2)[idx[3]]) + lnp[idx[3]])
        for i in range(3):
            ll += torch.sum(torch.log(p[idx[i], [i]]))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        # convert to floats / double
        if ctx.a.type() == 'torch.DoubleTensor':
            idx = [ctx.idx[i].double() for i in range(len(ctx.idx))]
            dlnLdp = torch.zeros(ctx.p.size()).double()
        else:
            idx = [ctx.idx[i].float() for i in range(len(ctx.idx))]
            dlnLdp = torch.zeros(ctx.p.size())

        # gradients for beta parameters
        dgab = torch.digamma(ctx.a + ctx.b)
        dlnLda = idx[3] * (torch.log(ctx.z) - torch.digamma(ctx.a) + dgab)
        dlnLdb = idx[3] * (torch.log(1-ctx.z) - torch.digamma(ctx.b) + dgab)

        # correct for accepts and rejects
        dlnLda[torch.isnan(dlnLda)] = 0
        dlnLdb[torch.isnan(dlnLdb)] = 0

        # gradients for discrete probabilities

        dlnLdp3 = idx[3] * torch.pow(1-torch.sum(ctx.p, 2), -1)
        for i in range(3):
            dlnLdp[:,:,i] = idx[i] * torch.pow(ctx.p[:,:,i], -1) - dlnLdp3

        # output
        outp = grad_output * -dlnLdp
        outa = grad_output * -dlnLda
        outb = grad_output * -dlnLdb

        return outp, outa, outb, None

