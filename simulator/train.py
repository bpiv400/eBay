import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from compress_pickle import load, dump
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize_scalar
from simulator.interface import Inputs, run_loop
from simulator.model import Simulator
from constants import *


class Trainer:
    def __init__(self, simulator, train, test):
        self.simulator = simulator
        self.train = train
        self.test = test
        self.model = self.simulator.model

    def init_training(self):
        # reset model parameters
        path = MODEL_DIR + 'pretrain/%s.net' % self.model
        self.simulator.net.load_state_dict(
            torch.load(path), strict=False)

        # initialize optimizer
        self.optimizer = optim.Adam(self.simulator.net.parameters())

        # count optimization iterations
        self.iter = 1

    def train_model(self, gamma):
        # reset model parameters to pretrained values and create optimizer
        self.init_training()

        # initialize tensorboard writer
        writer = SummaryWriter(
            LOG_DIR + '%s/%d' % (self.model, self.iter))

        # set gamma
        self.simulator.set_gamma(gamma)
        print('Iteration %d: gamma of %1.2f' % (self.iter, gamma))
        # training loop
        epoch, last = 0, np.inf
        while True:
            print('\tEpoch %d' % epoch)
            output = {}
            # train model
            t0 = dt.now()
            output['loss'] = run_loop(
                self.simulator, self.train, self.optimizer)
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
                if k in ['loss', 'penalty']:
                    print('\t\t%s: %d' % (k, v))
                elif 'lnL' in k:
                    print('\t\t%s: %1.4f' % (k, v))
                else:
                    print('\t\t%s: %2.2f' % (k, v))
            print('\t\ttime: %d sec' % (dt.now() - t0).total_seconds())

            # stopping condition: validation loss worsens
            if output['loss'] > FTOL * last:
                self.iter += 1
                writer.close()
                return output['lnL_test']

            # save model
            path = MODEL_DIR + '%s/%d_%d.net' \
                % (self.model, self.iter, epoch)
            torch.save(simulator.net.state_dict(), path)

            # increment epochs and update last
            epoch += 1
            last = output['loss']


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net and loss function
    simulator = Simulator(model, sizes)
    
    # print modules
    print(simulator.net)
    print(simulator.loss)

    # load datasets
    train = Inputs('train_models', model)
    test = Inputs('train_rl', model)

    # initialize trainer object
    trainer = Trainer(simulator, train, test)
    
    # find optimal gamma
    wrapper = lambda x: trainer.train_model(x)
    result = minimize_scalar(wrapper, 
        method='Bounded', bounds=(0,2), options={'xatol': 0.1})

    # save result
    dump(result, EXPS_DIR + '%s.pkl' % model)
    