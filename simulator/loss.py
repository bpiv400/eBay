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
	# for con_byr model, theta is a list
	if isinstance(theta, list):
		theta, theta4 = theta
		y, y4 = y

	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(num, torch.sum(num, dim=-1, keepdim=True))

	# loss is dot product of flow (p) and distance (y)
	loss = torch.sum(p * y)

	if torch.any(torch.isnan(loss)):
		print(theta)
		print(num)
		print(p)
		print(y)
		exit()

	# add in loss for 4th byr turn
	if 'theta4' in vars():
		p = torch.sigmoid(theta4)	# probability of accept
		errors = torch.pow(y4 - p, 2)
		loss += torch.sum(errors)

	return loss