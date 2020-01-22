import torch, torch.nn as nn


class VariationalDropout(nn.Module):
    def __init__(self, N):
        super(VariationalDropout, self).__init__()

        self.log_alpha = nn.Parameter(torch.Tensor(N))
        self.log_alpha.data.fill_(0)

    def kl_reg(self):
        # calculate KL divergence
        kl = 0.63576 * (torch.sigmoid(1.8732 + 1.48695 * self.log_alpha) - 1) \
            - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -torch.sum(kl)

    def forward(self, x):
        # additive N(0, alpha * x^2) noise
        if self.training:
            sigma = torch.sqrt(torch.exp(self.log_alpha)) * torch.abs(x)
            eps = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            return x + sigma * eps
        else:
            return x