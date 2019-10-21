from constants import *
import torch


def logit_loss(theta, y):
	p = torch.sigmoid(theta)
	ll = y * torch.log(p) + (1-y) * torch.log(1 - p)
	return -torch.sum(ll)


def poisson_loss(theta, y):
	# parameter
	l = torch.exp(theta)

	# log-likelihood
	ll = -l + y * torch.log(l)

	return -torch.sum(ll)


def emd_loss(theta, y, dim):
	# for con_byr model, theta is a list
	if isinstance(theta, list):
		theta, theta4 = theta

	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(theta, torch.sum(theta, dim=-1, keepdim=True))

	# earth mover's distance, broadcasting both y and dim
	dist = torch.pow(y.unsqueeze(dim=-1) - dim, 2)

	# loss is dot product of flow (p) and distance
	loss = torch.sum(p * dist)

	# add in loss for 4th byr turn
	if 'theta4' in vars():
		p = torch.sigmoid(theta4)	# probability of accept
		dist = torch.pow(y - p, 2)
		loss += torch.sum(dist)

	return loss