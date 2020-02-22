import numpy as np
from nets.FeedForward import FeedForward
from utils import load_sizes


class Model:
	def __init__(self, name, gamma=0, device='cuda'):
		# save parameters to self
		self.gamma = gamma
		self.device = device

		# boolean for penalty
		assert gamma >= 0
		self.penalized = gamma > 0

		# load model sizes
		sizes = load_sizes(name)

		# neural net
		self.net = FeedForward(sizes).to(device)

	def get_penalty(self, baserates):
		if self.gamma == 0:
			return 0

		penalty = 0.0
		for m in self.net.modules():
			if hasattr(m, 'kl_reg'):
				penalty += m.kl_reg().item()
		return self.gamma * penalty
