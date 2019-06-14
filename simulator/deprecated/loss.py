import torch
from torch.autograd import Function
from torch.distributions.beta import Beta
import numpy as np


def dlnLda(a, b, z, gamma, indices=None):
        '''
        Computes the derivative of the log of the beta pdf with respect to alpha.

        Inputs:
            - a (3, N, K): alpha parameters
            - b (3, N, K): beta parameters
            - z (3, N): observed outcome (i.e., delay or concession)
            - gamma (3, N, K): mixture weights
            - indices (3, N): 1.0 when 0 < z < 1

        Output:
            - da (3, N, K): derivative of lnL wrt alpha
        '''
        z = z.unsqueeze(dim=-1)
        da = torch.log(z) - torch.digamma(a) + torch.digamma(a + b)
        da *= gamma
        if indices is not None:
            da *= indices.unsqueeze(dim=-1)
            da[torch.isnan(da)] = 0
        return da


class SecLoss(Function):
    '''
    Computes the negative log-likelihood for the fully continuous
    beta mixture model.

    Inputs to forward:
        - a (mbsize, K): alpha parameter of beta distribution
        - b (mbsize, K): beta parameter of beta distribution
        - y (mbsize): continuous outcome
        - gamma (mbsize, K): class probabilities

    Output: negative log-likelihood
    '''
    @staticmethod
    def forward(ctx, a, b, y, gamma):
        # store variables for backward method
        ctx.a = a
        ctx.b = b
        ctx.y = y
        ctx.gamma = gamma

        # log-likelihood
        ll = torch.sum(torch.sum(Beta(a, b).log_prob(y.unsqueeze(dim=-1))))

        return -ll

    @staticmethod
    def backward(ctx, grad_output):
        # gradients for beta parameters
        da = dlnLda(ctx.a, ctx.b, ctx.y, ctx.gamma)
        db = dlnLda(ctx.b, ctx.a, 1-ctx.y, ctx.gamma)

        # output
        outa = grad_output * -da
        outb = grad_output * -db

        return outa, outb, None, None


class ConLoss(Function):
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

        # for gradcheck
        if ctx.a.type() == 'torch.DoubleTensor':
            ctx.idx = {key: val.double() for key, val in idx.items()}
        else:
            ctx.idx = {key: val.float() for key, val in idx.items()}

        # log-likelihood
        lndens = Beta(a, b).log_prob(y.unsqueeze(dim=-1))
        Q = torch.sum(gamma[idx[2]] * lndens[idx[2]], dim=1)
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