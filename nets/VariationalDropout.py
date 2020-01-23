import torch, torch.nn as nn
from nets.nets_consts import LNALPHA0, KL1, KL2, KL3


class VariationalDropout(nn.Module):
    def __init__(self, N):
        super(VariationalDropout, self).__init__()

        self.lnalpha = nn.Parameter(torch.Tensor(N))
        self.lnalpha.data.fill_(LNALPHA0)


    def kl_reg(self):
        # calculate KL divergence
        kl = KL1 * (torch.sigmoid(KL2 + KL3 * self.lnalpha) - 1) \
            - 0.5 * torch.log1p(torch.exp(-self.lnalpha))
        return -torch.sum(kl)


    def forward(self, x):
        # additive N(0, alpha * x^2) noise
        if self.training:
            sigma = torch.sqrt(torch.exp(self.lnalpha)) * torch.abs(x)
            eps = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            return x + sigma * eps
        else:
            return x