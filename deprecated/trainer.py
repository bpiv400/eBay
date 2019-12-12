import sys, math
import torch, torch.optim as optim
from datetime import datetime as dt
from simulator.interface import run_loop
from constants import *


class Trainer:
    def __init__(self, simulator, train, test, lr):
        # copy inputs to self
        self.simulator = simulator
        self.train = train
        self.test = test

        # initialize optimizer with default learning rate
        self.optimizer = optim.Adam(
            self.simulator.net.parameters(), lr=lr)
        print(self.optimizer)

    @staticmethod
    def weight_reset(m):
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()

    def reset_weights(self):
        self.simulator.net.apply(self.weight_reset)

    def train_model(self, log10_lr, writer, epochs):
    	print('Round %d' % self.iter)

        # reset model weights
        self.reset_weights()

        # training loop
        t0 = dt.now()
        lnL_train = run_loop(self.simulator, self.train, self.optimizer)
        print('\tTraining time: %d seconds' % \
        	(dt.now() - t0).total_seconds())
        print('\tlnL_train: %9.4f' % lnL_train)
        writer.add_scalar('lnL_train', lnL_train, self.iter)

        # calculate log-likelihood on validation set
        with torch.no_grad():
            t0 = dt.now()
            lnL_test = run_loop(self.simulator, self.test)
            print('\tValidation time: %d seconds' % \
            	(dt.now() - t0).total_seconds())
            print('\tlnL_test: %9.4f' % lnL_test)
            writer.add_scalar('lnL_test', lnL_test, self.iter)

        # save model
        torch.save(self.simulator.net.state_dict(), 
            MODEL_DIR + '%s/pretrain%d.net' % (self.simulator.model, self.iter))
