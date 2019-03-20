import torch
from torch.autograd import Function
import numpy as np
from utils import *


class BetaMixtureLoss(Function):
    '''
    Computes the negative log-likelihood for the beta mixture model.

    Inputs to forward:
        - p (N_turns, mbsize, M): discrete weights
        - a (N_turns, mbsize, K): alpha parameter of beta distribution
        - b (N_turns, mbsize, K): beta parameter of beta distribution
        - y (N_turns, mbsize): continuous outcome
        - gamma (N_turns, mbsize, K): class probabilities

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, p, a, b, y, gamma):
        # indices
        idx = {}
        idx[0] = y == 0
        idx[1] = y == 1
        idx[2] = ~idx[0] * ~idx[1] * ~torch.isnan(y)

        # store variables for backward method
        ctx.p = p
        ctx.a = a
        ctx.b = b
        ctx.y = y
        ctx.gamma = gamma
        if ctx.a.type() == 'torch.DoubleTensor':    # for gradcheck
            ctx.idx = {key: val.double() for key, val in idx.items()}
        else:
            ctx.idx = {key: val.float() for key, val in idx.items()}

        # log-likelihood
        lnp = ln_beta_pdf(a, b, y)
        Q = torch.sum(gamma[idx[2]] * lnp[idx[2]], dim=1)
        ll = torch.sum(torch.log(1 - torch.sum(p, 2)[idx[2]]) + Q)

        for i in range(p.size()[2]):
            ll += torch.sum(torch.log(p[idx[1-i], [i]]))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        # gradient for discrete probabilies
        p0 = 1 - torch.sum(ctx.p, dim=2, keepdim=True)
        psize = ctx.p.size()[2]
        inverse = torch.pow(p0, -1).repeat(1, 1, psize)
        dp = ctx.idx[2].unsqueeze(dim=2) * -inverse
        for i in range(psize):
            dp[:,:,i] += ctx.idx[1-i] * torch.pow(ctx.p[:,:,i], -1)

        # gradients for beta parameters
        da = dlnLda(ctx.a, ctx.b, ctx.y, ctx.gamma, ctx.idx[2])
        db = dlnLda(ctx.b, ctx.a, 1-ctx.y, ctx.gamma, ctx.idx[2])

        # output
        outp = grad_output * -dp
        outa = grad_output * -da
        outb = grad_output * -db

        return outp, outa, outb, None, None


class BinaryLoss(Function):
    '''
    Computes the negative log-likelihood for the slr msg model.

    Inputs to forward:
        - p (N_turns, mbsize, 1): probability of msg
        - y (N_turns, mbsize): indicator for msg

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, p, y):
        # indices
        idx = {}
        idx[1] = y == 1
        idx[0] = ~idx[1] * ~torch.isnan(y)

        # store variables for backward method
        ctx.p = p
        ctx.y = y
        if ctx.p.type() == 'torch.DoubleTensor':    # for gradcheck
            ctx.idx = {key: val.double() for key, val in idx.items()}
        else:
            ctx.idx = {key: val.float() for key, val in idx.items()}

        # log-likelihood
        ll = torch.sum(torch.log(1 - p[idx[0]]))
        ll += torch.sum(torch.log(p[idx[1]]))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        dp = - ctx.idx[0].unsqueeze(dim=2) * torch.pow(1-ctx.p, -1)
        dp += ctx.idx[1].unsqueeze(dim=2) * torch.pow(ctx.p, -1)

        return grad_output * -dp, None
