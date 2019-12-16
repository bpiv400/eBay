import sys, os, math
import torch, torch.optim as optim
import numpy as np
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from models.simulator.interface import run_loop
from models.simulator.model import Simulator
from constants import *


class Trainer:
    def __init__(self, model, sizes, train, test):
        self.model = model
        self.sizes = sizes
        self.train = train
        self.test = test
        self.iter = 0

        # attributes to be initialized later
        self.simulator = None
        self.loglr = None
        self.optimizer = None


    def init_training(self):
        # new instance of model
        self.simulator = Simulator(self.model, self.sizes, 
            dropout=self.iter > 0)  # after pretraining, use dropout
        if self.iter <= 1:
            print(self.simulator.net)

        # reset model parameters to pretrained parameters
        if self.iter > 0:
            path = MODEL_DIR + '%s/0.net' % self.model
            self.simulator.net.load_state_dict(
                torch.load(path), strict=False)

        # initialize learning rate and optimizer
        self.loglr = LOGLR0
        self.optimizer = optim.Adam(self.simulator.net.parameters(),
            lr=math.pow(10, self.loglr))
        print(self.optimizer)

    
    def train_model(self, gamma=0):
        # reset model parameters to pretrained values and create optimizer
        self.init_training()

        # initialize tensorboard writer
        writer = SummaryWriter(
            LOG_DIR + '%s/%d' % (self.model, self.iter))
        writer.add_scalar('gamma', gamma)

        # set gamma
        self.simulator.set_gamma(gamma)
        print('Iteration %d: gamma of %1.2f' % (self.iter, gamma))

        # training loop
        epoch, last = 0, np.inf
        while True:
            print('\tEpoch %d' % epoch)
            output = {'loglr': self.loglr}

            # train model
            t0 = dt.now()
            output['loss'] = run_loop(
                self.simulator, self.train, self.optimizer)

            # regularization stats
            if gamma > 0:
                output['penalty'] = self.simulator.get_penalty().item()
                output['share'], output['largest'] = \
                    self.simulator.get_alpha_stats()

            # calculate log-likelihood on training and validation sets
            with torch.no_grad():
                loss_train = run_loop(self.simulator, self.train)
                output['lnL_train'] = -loss_train / self.train.N_labels

                loss_test = run_loop(self.simulator, self.test)
                output['lnL_test'] = -loss_test / self.test.N_labels

            # save output to tensorboard writer and print to console
            for k, v in output.items():
                writer.add_scalar(k, v, epoch)
                if k in ['loss', 'penalty', 'largest', 'loglr']:
                    print('\t\t%s: %d' % (k, v))
                elif 'lnL' in k:
                    print('\t\t%s: %1.4f' % (k, v))
                else:
                    print('\t\t%s: %2.3f' % (k, v))
            print('\t\ttime: %d sec' % (dt.now() - t0).total_seconds())

            # reduce learning rate until convergence
            if output['loss'] > FTOL * last:
                if self.loglr == LOGLR1:
                    # save model
                    path = MODEL_DIR + '%s/%d.net' % (self.model, self.iter)
                    torch.save(self.simulator.net.state_dict(), path)

                    # increment iteration and close tensorboard writer
                    self.iter += 1
                    writer.close()

                    # return loss on holdout set 
                    return loss_test
                else:
                    self.loglr -= 1
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = math.pow(10, self.loglr)

            # increment epochs and update last
            epoch += 1
            last = output['loss']