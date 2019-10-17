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


def emd_loss(theta, y):
	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(theta, torch.sum(theta, dim=-1, keepdim=True))

	# earth mover's distance, broadcasting both y and CON_DIM
	dist = torch.pow(y - CON_DIM, 2)

	# loss is dot product of flow (p) and distance
	loss = torch.sum(p * dist)

	return loss