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


def emd_loss(theta, distance):
	# for con_byr model, theta is a list
	if isinstance(theta, list):
		theta, theta4 = theta
		distance, accept4 = distance

	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(num, torch.sum(num, dim=-1, keepdim=True))

	# loss is dot product of flow (p) and distance
	loss_i = torch.sum(p * distance, dim=1)
	loss = torch.sum(loss_i)

	# add in loss for 4th byr turn
	if 'theta4' in vars():
		p = torch.sigmoid(theta4)	# probability of accept
		errors = torch.pow(accept4 - p, 2)
		loss += torch.sum(errors)

	return loss

def cross_entropy_loss(theta, y):
	# for con_byr model, theta is a list
	if isinstance(theta, list):
		theta, theta4 = theta
		y, y4 = y

	# predicted bucket probabilities
	num = torch.exp(theta)
	p = torch.div(num, torch.sum(num, dim=-1, keepdim=True))

	# loss is dot product of flow (p) and distance
	loss = torch.sum(torch.log(p[y]))

	# add in loss for 4th byr turn
	if 'theta4' in vars():
		p4 = torch.sigmoid(theta4)	# probability of accept
		loss += torch.sum(torch.log(p4[y4]))

	return loss