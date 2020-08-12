import torch
import numpy as np
from scipy.special import betainc


def beta_cdf(a, b, x, n_iter=41):
    beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    # split values
    S0 = (a + 1) / (a + b + 2)
    x_l = torch.min(x, S0)
    x_h = torch.max(x, S0)

    # low values
    T = torch.zeros_like(x)
    for k in reversed(range(n_iter)):
        v = -(a + k) * (a + b + k) * x_l / (a + 2 * k) / (a + 2 * k + 1)
        T = v / (T + 1)
        if k > 0:
            v = k * (b - k) * x_l / (a + 2 * k - 1) / (a + 2 * k)
            T = v / (T + 1)

    cdf_l = x_l ** a * (1 - x_l) ** b / (a * beta) / (T + 1)

    # high values
    T = torch.zeros_like(x)
    for k in reversed(range(n_iter)):
        v = -(b + k) * (a + b + k) * (1 - x_h) / (b + 2 * k) / (b + 2 * k + 1)
        T = v / (T + 1)
        if k > 0:
            v = k * (a - k) * (1 - x_h) / (b + 2 * k - 1) / (b + 2 * k)
            T = v / (T + 1)

    cdf_h = 1 - x_h ** a * (1 - x_h) ** b / (b * beta) / (T + 1)

    # concatenate
    return cdf_l * (x <= S0) + cdf_h * (x > S0)


def cdf0(a, b, x):
    beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    S0 = (a + 1) / (a + b + 2)
    DK = torch.zeros(21)
    if x <= S0:
        for k in range(1, 11):
            DK[2 * k - 1] = k * (b - k) * x / (a + 2 * k - 1) / (a + 2 * k)
        for k in range(11):
            DK[2 * k] = -(a + k) * (a + b + k) * x / (a + 2 * k) / (a + 2 * k + 1)
        T1 = 0.0
        for i in reversed(range(21)):
            T1 = DK[i] / (T1+1)
        return x ** a * (1-x) ** b / (a * beta) / (1+T1)


def main():
    a = torch.tensor([.2, 100., 2.5]).unsqueeze(-1)
    b = torch.tensor([10., .3, 2.5]).unsqueeze(-1)
    dim = torch.from_numpy(np.linspace(0, 1, 102)).float()
    x = dim.expand(a.size()[0], -1)

    scipy = betainc(a, b, x)
    copy = beta_cdf(a, b, x)
    diff = scipy - copy
    print('Scipy: {}'.format(scipy))
    print('Copy: {}'.format(copy))
    print('Diff: {}'.format(diff))
    print('Max diff: {}'.format(torch.max(torch.abs(diff)).item()))


if __name__ == '__main__':
    main()
