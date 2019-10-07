from constants import *
from torch.distributions.beta import Beta
import torch


def logit_loss(theta, y):
	p = torch.sigmoid(theta.squeeze())
	ll = y * torch.log(p) + (1-y) * torch.log(1 - p)
	return -torch.sum(ll), None


def beta_mixture_loss(theta, y, omega=None):
	# exponentiate
	theta = torch.exp(theta.squeeze())

	# parse parameters
	k = int(theta.size()[-1] / 3)
	a = 1 + torch.index_select(theta, -1, torch.tensor(range(k), device=DEVICE))
	b = 1 + torch.index_select(theta, -1, torch.tensor(range(k, 2 * k), device=DEVICE))
	c = torch.index_select(theta, -1, torch.tensor(range(2 * k, 3 * k), device=DEVICE))

	# beta densities
	lndens = Beta(a, b).log_prob(y.unsqueeze(dim=-1))

	# multinomial probabilities
	phi = torch.div(c, torch.sum(c, dim=-1, keepdim=True))

	# calculate weights
	if omega is None:
		dens = torch.exp(lndens.detach())
		omega = torch.div(dens, torch.sum(dens, dim=-1, keepdim=True))

	# expected log-likelihood
	ll = (lndens + torch.log(phi) - torch.log(omega))
	Q = torch.sum(omega * ll)

	# calculate new weights
	dens = torch.exp(lndens.detach())
	omega = torch.div(dens, torch.sum(dens, dim=-1, keepdim=True))

	# calculate negative log-likelihood
	return -Q, omega


def negative_binomial_loss(theta, y):
	theta = theta.squeeze()
	# parameters
	r = torch.exp(torch.index_select(theta, -1, torch.tensor(0))).squeeze()
	p = torch.sigmoid(torch.index_select(theta, -1, torch.tensor(1))).squeeze()
	# log-likelihood components
	ll = torch.mvlgamma(y + r, 1) - torch.mvlgamma(r, 1)
	ll += y * torch.log(p) + r * torch.log(1 - p)
	return -torch.sum(ll), None

