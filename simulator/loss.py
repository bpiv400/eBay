import torch
from torch.distributions.beta import Beta


def LogitLoss(theta, y):
	p = torch.sigmoid(theta)
	ll = y * torch.log(p) + (1-y) * torch.log(1-p)
	return -torch.sum(ll)


def BetaMixtureLoss(theta, omega, y):
    # exponentiate
	theta = torch.exp(theta)

    # parse parameters
	K = int(theta.size()[-1] / 3)
	a = 1 + torch.index_select(theta, -1, torch.tensor(range(K)))
	b = 1 + torch.index_select(theta, -1, torch.tensor(range(K, 2 * K)))
	c = torch.index_select(theta, -1, torch.tensor(range(2 * K, 3 * K)))

    # beta densities
	lndens = Beta(a, b).log_prob(y.unsqueeze(dim=-1))

	# multinomial probabilities
	phi = torch.div(c, torch.sum(c, dim=-1, keepdim=True))

	# expected log-likelihood
	ll = (lndens + torch.log(phi) - torch.log(omega))
	Q = torch.sum(omega * ll)

	# calculate new weights
	dens = torch.exp(lndens.detach())
	w = torch.div(dens, torch.sum(dens, dim=-1, keepdim=True))

    # calculate negative log-likelihood
	return -Q, w


def NegativeBinomialLoss(theta, y):
    # parameters
    r = torch.exp(torch.index_select(theta, -1,
        torch.tensor(0))).squeeze()
    p = torch.sigmoid(torch.index_select(theta, -1,
        torch.tensor(1))).squeeze()
    # log-likelihood components
    ll = torch.mvlgamma(y + r, 1) - torch.mvlgamma(r, 1)
    ll += y * torch.log(p) + r * torch.log(1-p)
    return -torch.sum(ll)

