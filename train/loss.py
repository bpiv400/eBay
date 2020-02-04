import torch
from torch.nn.functional import log_softmax


def time_loss(theta, y):
	# class probabilities
	lnp = log_softmax(theta, dim=-1)

	# arrivals have positive y
	arrival = y >= 0
	lnL = torch.sum(lnp[arrival, y[arrival]])

	# non-arrivals
	cens = y < 0
	y_cens = y[cens]
	p_cens = torch.exp(lnp[cens, :])
	for i in range(p_cens.size()[0]):
		lnL += torch.log(torch.sum(p_cens[i, y_cens[i]:]))

	return -lnL
