import numpy as np
import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from compress_pickle import load
from model.Model import Model
from model.datasets.eBayDataset import eBayDataset
from model.model_consts import LOG_DIR, LOGLR0, LOGLR1, LOGLR_INC, FTOL
from constants import INPUT_DIR, MODEL_DIR, PARAMS_PATH


class Optimizer:
	def __init__(self, params):

		self.optimizer = optim.Adam(params)
		self.loglr = None
		self.reset()

	def reset(self):
		self.set_loglr(LOGLR0)

    def set_loglr(self, loglr):
		self.loglr = loglr
		for param_group in optimizer.param_groups:
			param_group['lr'] = 10 ** loglr


class Trainer:
	'''
	Trains a model until convergence.

	Public methods:
		* train: trains the initialized model under given parameters.
	'''
	def __init__(self, name, train_part, test_part):
		'''
		:param name: string model name.
		:param train_part: string partition name for training data.
		:param test_part: string partition name for holdout data.
		'''
		# save name to self
		self.name = name

		# load model sizes
	    sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
	    print('Sizes: {}'.format(sizes))

	    # load parameters
	    params = load(PARAMS_PATH)
	    print('Parameters: {}'.format(params))

	    # initialize model
	    self.model = Model(name, sizes, params)
	    print(self.model.net)
	    print(self.model.loss.__name__)

	    # load datasets
	    self.train = eBayDataset(train_part, name)
	    self.test = eBayDataset(test_part, name)

		# initialize optimizer
	    self.optimizer = Optimizer(self.model.net.parameters())

		# experiment id
		self.expid = 0

	def train(self, gamma=0, smoothing=0):
		'''
		Public method to train model.
		:param gamma: scalar regularization parameter for variational dropout.
		:param smoothing: scalar smoothing parameter for arrival/delay models.
		'''
		# reset optimizer
		self.optimizer.reset()

		# save hyperparameters in model
		self.model.gamma = gamma
		self.model.smoothing = smoothing

	    # initialize writer
		writer = SummaryWriter(
			LOG_DIR + '{}/{}'.format(self.name, self.expid))

		# training loop
	    epoch, last, loglr = 0, np.inf, LOGLR0
	    while True:
	        print('\tEpoch %d' % epoch)

	        # run one epoch
	        output = self._run_epoch()

	        # save output to tensorboard writer
	        for k, v in output.items():
	            writer.add_scalar(k, v, epoch)

	        # save model and clean up
        	torch.save(self.model.net.state_dict(), 
        		MODEL_DIR + '{}/{}.net'.format(self.name, self.expid))

	        # reduce learning rate until convergence
        	if output['loss'] > FTOL * last:
	            if self.optimizer.loglr == LOGLR1:
	                break
	            else:
	                self.optimizer.set_loglr(
	                	self.optimizer.loglr - LOGLR_INC)

	        # update last, increment epoch
	        last = output['loss']
	        epoch += 1

	    # increment expid and clean up
	    self.expid += 1
	    writer.close()

    def _run_epoch(self):
	 	# initialize output with log10 learning rate
		output = {'loglr': self.optimizer.loglr,
				  'gamma': self.model.gamma,
				  'smoothing': self.model.smoothing}

        # train model
        output['loss'] = self.model.run_loop(self.train, self.optimizer)
        output['lnL_train'] = -output['loss'] / self.train.N

        # calculate log-likelihood on validation set
        with torch.no_grad():
            loss_test = self.model.run_loop(self.test)
            output['lnL_test'] = -loss_test / self.test.N

        return output['loss']




