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
	k = int(theta.size()[-1] / 2)
	a = 1 + torch.index_select(theta, -1, 
		torch.tensor(range(k), device=DEVICE))
	b = 1 + torch.index_select(theta, -1,
		torch.tensor(range(k, 2 * k), device=DEVICE))

	# beta densities
	lndens = Beta(a, b).log_prob(y.unsqueeze(dim=-1))
	dens = torch.exp(lndens.detach())

	# calculate weights
	if omega is None:
		omega = torch.div(dens, torch.sum(dens, dim=-1, keepdim=True))

	# expected log-likelihood
	Q = torch.sum(omega * lndens)

	# calculate new weights
	omega = torch.div(dens, torch.sum(dens, dim=-1, keepdim=True))

	# calculate negative log-likelihood
	return -Q, omega


def negative_binomial_loss(theta, y):
	theta = theta.squeeze()
	# parameters
	r = torch.exp(torch.index_select(theta, -1, 
		torch.tensor(0, device=DEVICE))).squeeze()
	p = torch.sigmoid(torch.index_select(theta, -1, 
		torch.tensor(1, device=DEVICE))).squeeze()
	# log-likelihood components
	ll = torch.mvlgamma(y + r, 1) - torch.mvlgamma(r, 1)
	ll += y * torch.log(p) + r * torch.log(1 - p)

	return -torch.sum(ll), None


def poisson_loss(theta, y):
	# parameter
	l = torch.exp(theta.squeeze())

	# log-likelihood
	ll = -l + y * torch.log(l)

	return -torch.sum(ll), None