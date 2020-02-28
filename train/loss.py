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


def taylor_softmax_loss(theta, y):
	num = theta + (theta ** 2) / 2 + 1
	den = torch.sum(num, -1)
	lnp = torch.log(torch.take(num, y)) - torch.log(den)
	lnL = torch.sum(lnp)

	return -lnL


def sparsemax_tau(z):
	"""
	:param z: Input tensor. First dimension should be the batch size
	:return torch.Tensor: [batch_size x 1] Output tensor
	"""
	K = z.size()[-1]

	# Translate input by max for numerical stability
	z = z - torch.max(z, dim=-1, keepdim=True)[0]

	# Sort input in descending order and take cumulative sum
	z_sorted = torch.sort(z, descending=True)[0]
	z_sum = torch.cumsum(z_sorted, -1)

	# Determine sparsity of projection
	ar = torch.arange(start=1, end=K + 1, step=1, device=z.device, dtype=z.dtype).view(1, -1)
	bound = 1 + z_sorted * ar
	is_gt = torch.gt(bound, z_sum).type(z.type())
	k = torch.max(is_gt * ar, -1, keepdim=True)[0]

	# Compute threshold function
	z_sparse = is_gt * z_sorted

	# Compute taus
	tau = (torch.sum(z_sparse, -1, keepdim=True) - 1) / k

	return tau, z_sparse, k.long()


def sparsemax_p(z):
	"""
	:param z: Input tensor. First dimension should be the batch size
	:return torch.Tensor: [batch_size x number_of_logits] Output tensor
	"""
	# sparsemax taus
	tau, _, _ = sparsemax_tau(z)

	# sparsemax probabilities
	p = torch.max(torch.zeros_like(z), z - tau)

	return p


def sparsemax_loss(z, y):
	# model output for golden key
	z_k = torch.take(z, y)

	# difference between z^2 and tau^2
	tau, z_sparse, k = sparsemax_tau(z)
	obs = tau.size()[0]
	diff = torch.zeros(obs, device=z.device, dtype=z.dtype)
	for i in range(obs):
		z2 = z_sparse[i, :k[i]] ** 2
		tau2 = tau[i] ** 2
		diff[i] = torch.sum(z2 - tau2)

	# loss
	half = torch.tensor(0.5, device=z.device, dtype=z.dtype)
	loss = torch.sum(half - z_k + half * diff)

	return loss
