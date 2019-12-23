import sys, os, math
import torch, torch.optim as optim
import numpy as np
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from models.Model import Model
from constants import *


class Trainer:
    def __init__(self, name, sizes, train, test):
        self.name = name
        self.sizes = sizes
        self.train = train
        self.test = test

        # start with pretrained model if it exists
        self.pretrained_path = MODEL_DIR + '%s/0.net' % name
        if os.path.isfile(self.pretrained_path):
            self.iter = 1
        else:
            self.iter = 0

        # attributes to be initialized later
        self.model = None
        self.loglr = None
        self.optimizer = None

    
    def train_model(self, gamma=0):
        # reset model parameters to pretrained values and create optimizer
        self._init_training()

        # initialize tensorboard writer
        writer = SummaryWriter(
            LOG_DIR + '%s/%d' % (self.name, self.iter))
        
        # set gamma
        if gamma > 0:
            self.model.set_gamma(gamma)
            writer.add_scalar('gamma', gamma)
            print('Iteration %d: gamma of %1.2f' % (self.iter, gamma))

        # model path
        model_path = MODEL_DIR + '%s/%d.net' % (self.name, self.iter)

        # training loop
        epoch, last = 0, np.inf
        while True:
            print('\tEpoch %d' % epoch)
            output = {'loglr': self.loglr}

            # train model
            t0 = dt.now()
            output['loss'] = self.model.run_loop(self.train, self.optimizer)

            # regularization stats
            if gamma > 0:
                output['penalty'] = self.model.get_penalty().item()
                output['share'], output['largest'] = \
                    self.model.get_alpha_stats()

            # calculate log-likelihood on training and validation sets
            with torch.no_grad():
                loss_train = self.model.run_loop(self.train)
                output['lnL_train'] = -loss_train / self.train.N_labels

                loss_test = self.model.run_loop(self.test)
                output['lnL_test'] = -loss_test / self.test.N_labels

            # save output to tensorboard writer and print to console
            for k, v in output.items():
                writer.add_scalar(k, v, epoch)
                if k in ['loss', 'penalty', 'largest']:
                    print('\t\t%s: %d' % (k, v))
                else:
                    print('\t\t%s: %1.4f' % (k, v))
            print('\t\ttime: %d sec' % (dt.now() - t0).total_seconds())

            # save model
            torch.save(self.model.net.state_dict(), model_path)

            # reduce learning rate until convergence
            if output['loss'] > FTOL * last:
                if self.loglr == LOGLR1:
                    self.iter += 1
                    writer.close()
                    return loss_test
                else:
                    self.loglr -= LOGLR_INC
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = math.pow(10, self.loglr)

            # increment epochs and update last
            epoch += 1
            last = output['loss']


    def _init_training(self):
        # new instance of model
        self.model = Model(self.name, self.sizes, 
            dropout=self.iter > 0)  # after pretraining, use dropout
        if self.iter <= 1:
            print(self.model.net)

        # reset model parameters to pretrained parameters
        if self.iter > 0:
            self.model.net.load_state_dict(
                torch.load(self.pretrained_path), strict=False)

        # initialize learning rate and optimizer
        self.loglr = LOGLR0
        self.optimizer = optim.Adam(self.model.net.parameters(),
            lr=math.pow(10, self.loglr))
        print(self.optimizer)