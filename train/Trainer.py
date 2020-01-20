import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from compress_pickle import load
from model.nets import FeedForward
from train.eBayDataset import eBayDataset
from train.train_consts import FTOL, LOG_DIR
from train.Sample import get_batches
from train.Optimizer import Optimizer
from constants import INPUT_DIR, MODEL_DIR, PARAMS_PATH


class Trainer:
	'''
	Trains a model until convergence.

	Public methods:
		* train: trains the initialized model under given parameters.
	'''
	def __init__(self, name, train_part, test_part, expid=None, device='cuda'):
		'''
		:param name: string model name.
		:param train_part: string partition name for training data.
		:param test_part: string partition name for holdout data.
		:param expid: string id of experiment for log and model filenames.
		'''
		# save parameters to self
		self.name = name
		self.expid = expid
		self.device = device

		# universal epoch
		self.epoch = 0

		# load model sizes
		sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
		print('Sizes: {}'.format(sizes))

		# load parameters
		params = load(PARAMS_PATH)
		print('Parameters: {}'.format(params))

		# initialize regularization hyperparameters to 0
		self.gamma = 0.0
		self.smoothing = 0.0

		# loss function
		if name in ['hist', 'con_slr', 'con_byr']:
			self.loss = nn.CrossEntropyLoss(reduction='sum')
		elif 'msg' in name:
			self.loss = nn.BCEWithLogitsLoss(reduction='sum')
		else:
			self.loss = self._TimeLoss

		# neural net
		self.net = FeedForward(sizes, params).to(device)
		print(self.net)

		# load datasets
		self.train = eBayDataset(train_part, name)
		self.test = eBayDataset(test_part, name)

		# set expid to None for development
		if self.expid is not None:
			# initialize writer
			self.writer = SummaryWriter(
				LOG_DIR + '{}/{}'.format(name, expid))

			# train for one epoch with penalty hyperparameters set to 0
			self.pretrained_path = MODEL_DIR + '{}/pretrained.net'.format(name)
			if not os.path.isfile(self.pretrained_path):
				self._run_epoch()
				torch.save(self.net.state_dict(), self.pretrained_path)


	def train_model(self, gamma=0, smoothing=0):
		'''
		Public method to train model.
		:param gamma: scalar regularization parameter for variational dropout.
		:param smoothing: scalar smoothing parameter for arrival/delay models.
		'''
		# initialize optimizer
		optimizer = Optimizer(self.net.parameters())
		print(optimizer.optimizer)

		# save hyperparameters in model and to writer
		self.gamma = gamma
		self.smoothing = smoothing

		# load pre-trained model weights
		if self.expid is not None:
			self.net.load_state_dict(
				torch.load(self.pretrained_path))

		# training loop
		last = np.inf
		while True:
			# run one epoch
			output = self._run_epoch(optimizer)

			# save model
			if self.expid is not None:
				torch.save(self.net.state_dict(), 
					MODEL_DIR + '{}/{}.net'.format(self.name, self.expid))

			# reduce learning rate until convergence
			if output['loss'] > FTOL * last:
				if optimizer.check():
					break
				else:
					optimizer.step()

			# update last, increment epoch
			last = output['loss']
			self.epoch += 1

		return -output['lnL_test']


	def _run_epoch(self, optimizer):
		print('\tEpoch %d' % self.epoch)

	 	# initialize output with log10 learning rate
		output = {'loglr': optimizer.loglr,
				  'gamma': self.gamma,
				  'smoothing': self.smoothing}

		# train model
		output['loss'] = self._run_loop(
			self.train, optimizer.optimizer)
		output['lnL_train'] = -output['loss'] / self.train.N

		# variational dropout stats
		output['share'], output['largest'] = self._get_alpha_stats()

		# calculate log-likelihood on validation set
		with torch.no_grad():
			loss_train = self._run_loop(self.train)
			output['lnL_train'] = -loss_train / self.train.N
			loss_test = self._run_loop(self.test)
			output['lnL_test'] = -loss_test / self.test.N

		# save output to tensorboard writer
		for k, v in output.items():
			print('\t\t{}: {}'.format(k, v))
			if self.expid is not None:
				self.writer.add_scalar(k, v, self.epoch)

		return output


	def _run_loop(self, data, optimizer=None):
		'''
		Calculates loss on epoch, steps down gradients if training.
		:param data: Inputs object.
		:param optimizer: instance of torch.optim.
		:return: scalar loss.
		'''
		batches = get_batches(data, 
		    isTraining=optimizer is not None)

		# loop over batches, calculate log-likelihood
		loss = 0.0
		gpu_time = 0.0
		t0 = dt.now()
		for b in batches:
			t1 = dt.now()

			# move to device
			b['x'] = {k: v.to(self.device) for k, v in b['x'].items()}
			b['y'] = b['y'].to(self.device)

			# increment loss
			loss += self._run_batch(b, optimizer)

			# increment gpu time
			gpu_time += (dt.now() - t1).total_seconds()

		print('\t\tGPU time: {0:.1f} seconds'.format(gpu_time))
		print('\t\tTotal time: {0:.1f} seconds'.format(
			(dt.now() - t0).total_seconds()))

		return loss


	def _run_batch(self, b, optimizer):
		'''
		Loops over examples in batch, calculates loss.
		:param b: batch of examples from DataLoader.
		:param optimizer: instance of torch.optim.
		:return: scalar loss.
		'''
		isTraining = optimizer is not None  # train / eval mode

		# call forward on model
		self.net.train(isTraining)
		theta = self.net(b['x'])

		# binary cross entropy requires float
		if str(self.loss) == "BCEWithLogitsLoss()":
			b['y'] = b['y'].float()

		# calculate loss
		loss = self.loss(theta.squeeze(), b['y'].squeeze())

		# add in regularization penalty and step down gradients
		if isTraining:
			if self.gamma > 0:
				loss = loss + self._get_penalty(len(b['y']))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		return loss.item()


	def _TimeLoss(self, theta, y):
		# class probabilities
		lnp = nn.functional.log_softmax(theta, dim=-1)

		# arrivals have positive y
		arrival = y >= 0
		lnL = torch.sum(lnp[arrival, y[arrival]])

		# non-arrivals
		cens = y < 0
		y_cens = y[cens]
		p_cens = torch.exp(lnp[cens, :])
		for i in range(p_cens.size()[0]):
			lnL += torch.log(torch.sum(p_cens[i, y_cens[i]:]))

		# penalty for jumps between classes
		if self.net.training and self.smoothing > 0.0:
			p = torch.exp(lnp)[:, :-1]  # ignore last index (expiration)
			jumps = p[:, 1:] - p[:, :-1]
			penalty = self.smoothing * torch.sum(jumps ** 2)
		else:
			penalty = 0

		return -lnL + penalty


	def _get_penalty(self, mbsize):
		penalty = 0.0
		for m in self.net.modules():
			if hasattr(m, 'kl_reg'):
				penalty += m.kl_reg().item()
		return self.gamma * penalty / mbsize


	def _get_alpha_stats(self, threshold=9):
		above, total, largest = 0.0, 0.0, 0.0
		for m in self.net.modules():
			if hasattr(m, 'log_alpha'):
				alpha = torch.exp(m.log_alpha)
				largest = max(largest, torch.max(alpha).item())
				total += alpha.size()[0]
				above += torch.sum(alpha > threshold).item()
		return above / total, largest
