import torch
import torch.nn as nn
from compress_pickle import load
from model.nets import FeedForward
from train.train_consts import LOGLR0, LOGLR1, LOGLR_INC
from constants import INPUT_DIR, PARAMS_PATH


class Model:
	def __init__(self, name, gamma=0, device='cuda'):
		# save parameters to self
		self.name = name
		self.gamma = gamma
		self.device = device
		
		# dropout flag
		self.dropout = gamma > 0

		# load model sizes and parameters
		sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
		params = load(PARAMS_PATH)

		# neural net
		self.net = FeedForward(sizes, params, dropout=self.dropout).to(device)


	def get_penalty(self):
		if self.gamma == 0:
			return 0

		penalty = 0.0
		for m in self.net.modules():
			if hasattr(m, 'kl_reg'):
				penalty += m.kl_reg().item()
		return self.gamma * penalty


	def get_alpha_stats(self):
		m1, m2, N, largest = 0.0, 0.0, 0, 0.0
		for m in self.net.modules():
			if hasattr(m, 'log_alpha'):
				alpha = torch.exp(m.log_alpha)
				largest = max(largest, torch.max(alpha).item())
				m1 += torch.sum(alpha).item()
				m2 += torch.sum(alpha ** 2).item()
				N += len(alpha)

		std = (m2/N - (m1/N) ** 2) ** 0.5
		return std, largest
