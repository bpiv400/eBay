import torch
import torch.nn as nn
import numpy as np
from compress_pickle import load
from nets.FeedForward import FeedForward
from constants import INPUT_DIR
from utils import load_sizes


class Model:
	def __init__(self, name, gamma=0, device='cuda'):
		# save parameters to self
		self.gamma = gamma
		self.device = device
		
		# dropout flag
		self.dropout = gamma > 0

		# load model sizes
		sizes = load_sizes(name)

		# neural net
		self.net = FeedForward(sizes, dropout=self.dropout).to(device)

	def get_penalty(self):
		if self.gamma == 0:
			return 0

		penalty = 0.0
		for m in self.net.modules():
			if hasattr(m, 'kl_reg'):
				penalty += m.kl_reg().item()
		return self.gamma * penalty

	def get_lnalpha_stats(self):
		lnalpha = self.lnalpha
		return np.std(lnalpha), np.max(lnalpha)

	@property
	def lnalpha(self):
		lnalpha = []
		for m in self.net.modules():
			if hasattr(m, 'log_alpha'):
				lnalpha.append(m.log_alpha.detach().cpu().numpy())
		return np.concatenate(lnalpha)
