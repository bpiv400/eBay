import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from compress_pickle import load
from train.eBayDataset import eBayDataset
from train.train_consts import FTOL, LOG_DIR
from train.Model import Model
from train.Sample import get_batches
from train.Optimizer import Optimizer
from train.TimeLoss import TimeLoss
from constants import MODEL_DIR


class Trainer:
	'''
	Trains a model until convergence.

	Public methods:
		* train: trains the initialized model under given parameters.
	'''
	def __init__(self, name, train_part, test_part, dev=False, device='cuda'):
		'''
		:param name: string model name.
		:param train_part: string partition name for training data.
		:param test_part: string partition name for holdout data.
		:param dev: True for development.
		'''
		# save parameters to self
		self.name = name
		self.dev = dev
		self.device = device

		# path to pretrained model
		self.pretrained_path = MODEL_DIR + '{}/pretrained.net'.format(name)

		# load datasets
		self.train = eBayDataset(train_part, name)
		self.test = eBayDataset(test_part, name)

		# loss function
		if name in ['hist', 'con_slr', 'con_byr']:
			self.loss = nn.CrossEntropyLoss(reduction='sum')
		elif 'msg' in name:
			self.loss = nn.BCEWithLogitsLoss(reduction='sum')
		else:
			self.loss = TimeLoss

		# model and writer to be initialized in training loop
		self.model = None
		self.writer = None
		
		# pretrain with penalty hyperparameter set to 0
		if not os.path.isfile(self.pretrained_path):
			self._pretrain()


	def _pretrain(self):
		# neural net without dropout
		self.model = Model(self.name, gamma=0, device=self.device)
		print(self.model.net)

		# train without dropout until convergence
		print('Pretraining:')
		self.train_model()

		# save pretrained model
		torch.save(self.model.net.state_dict(), self.pretrained_path)


	def train_model(self, gamma=0):
		'''
		Public method to train model.
		:param gamma: scalar regularization parameter for variational dropout.
		'''
		# epoch number
		epoch = 0

		# initialize model
		self.model = Model(self.name, gamma=gamma, device=self.device)
		print(self.model.net)

		# load pre-trained model weights
		if self.model.dropout:
			self.model.net.load_state_dict(
				torch.load(self.pretrained_path), strict=False)

		# initialize optimizer
		optimizer = Optimizer(self.model.net.parameters())
		print(optimizer.optimizer)

		# initialize writer
		if not self.dev:
			expid = dt.now().strftime('%y%m%d-%H%M')
			self.writer = SummaryWriter(
				LOG_DIR + '{}/{}'.format(self.name, expid))

		# training loop
		last = np.inf
		while True:
			# run one epoch
			print('Epoch {}'.format(epoch))
			output = self._run_epoch(optimizer, epoch=epoch)

			# save model
			if not self.dev:
				torch.save(self.net.state_dict(), 
					MODEL_DIR + '{}/{}.net'.format(self.name, expid))

			# reduce learning rate until convergence
			if output['loss'] > FTOL * last:
				if optimizer.check():
					break
				else:
					optimizer.step()

			# update last, increment epoch
			last = output['loss']
			epoch += 1

		return -output['lnL_test']


	def _run_epoch(self, optimizer, epoch=None):
	 	# initialize output with log10 learning rate
		output = {'loglr': optimizer.loglr}

		# train model
		output['loss'] = self._run_loop(
			self.train, optimizer.optimizer)
		output['lnL_train'] = -output['loss'] / self.train.N

		# variational dropout stats
		if self.model.dropout:
			output['gamma'] = self.model.gamma
			output['std_alpha'], output['max_alpha'] = \
				self.model.get_alpha_stats()

		# calculate log-likelihood on validation set
		with torch.no_grad():
			loss_train = self._run_loop(self.train)
			output['lnL_train'] = -loss_train / self.train.N
			loss_test = self._run_loop(self.test)
			output['lnL_test'] = -loss_test / self.test.N

		# save output to tensorboard writer
		for k, v in output.items():
			print('\t{}: {}'.format(k, v))
			if self.writer is not None:
				self.writer.add_scalar(k, v, epoch)

		return output


	def _run_loop(self, data, optimizer=None):
		'''
		Calculates loss on epoch, steps down gradients if training.
		:param data: Inputs object.
		:param optimizer: instance of torch.optim.
		:return: scalar loss.
		'''
		isTraining = optimizer is not None
		batches = get_batches(data, isTraining=isTraining)

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

		# print timers
		prefix = 'training' if isTraining else 'validation'
		print('\t{0:s} GPU time: {1:.1f} seconds'.format(prefix, gpu_time))
		print('\ttotal {0:s} time: {1:.1f} seconds'.format(prefix, 
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
		self.model.net.train(isTraining)
		theta = self.model.net(b['x'])

		# binary cross entropy requires float
		if str(self.loss) == "BCEWithLogitsLoss()":
			b['y'] = b['y'].float()

		# calculate loss
		loss = self.loss(theta.squeeze(), b['y'].squeeze())

		# add in regularization penalty and step down gradients
		if isTraining:
			if self.model.dropout:
				penalty = self.model.get_penalty()
				factor = len(b['y']) / len(self.train)
				#print(loss.item(), penalty * factor)
				loss = loss + penalty * factor

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		return loss.item()
