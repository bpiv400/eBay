from torch import optim
from train.train_consts import LOGLR0, LOGLR1, LOGLR_INC


class Optimizer:
	def __init__(self, params):
		self.optimizer = optim.Adam(params)
		self.loglr = None
		self.reset()

	def reset(self):
		self._set_loglr(LOGLR0)

	def step(self):
		self._set_loglr(self.loglr - LOGLR_INC)

	def check(self):
		return self.loglr <= LOGLR1

	def _set_loglr(self, loglr):
		self.loglr = loglr
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = 10 ** self.loglr