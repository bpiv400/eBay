from constants import *
import torch


def logit_loss(theta, y):
	p = torch.sigmoid(theta)	# predicted probability
	ll = y * torch.log(p) + (1-y) * torch.log(1 - p)
	return -torch.sum(ll)


def poisson_loss(theta, y):
	l = torch.exp(theta)	# lambda parameter
	ll = -l + y * torch.log(l)	# log-likelihood
	return -torch.sum(ll)


def cross_entropy_loss(theta, y):
	# for con_byr model, theta is a list
	if isinstance(theta, list):
		theta, theta4 = theta
		y, y4 = y

	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(num, torch.sum(num, dim=-1, keepdim=True))

	# log-likelihood
	ll = torch.log(torch.gather(p, 1, y.unsqueeze(dim=1).long()))

	# add in logistic log-likelihood for 4th byr turn
	if 'theta4' in vars():
		p4 = torch.sigmoid(theta4)	# probability of accept
		y4 = (y4 == 100).unsqueeze(dim-1)
		ll += y4 * torch.log(p4) + (1-y4) * torch.log(1-p4)

	return -torch.sum(ll)