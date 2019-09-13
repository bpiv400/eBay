import sys, math
sys.path.append('../')
from constants import *
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions.beta import Beta
from torch.nn.utils import rnn
import numpy as np
from loss import *
from nets import *


class Simulator:
    '''
    Constructs neural network and holds gamma for beta mixture model.
    '''
    def __init__(self, model, outcome, train, params, sizes):
        # initialize parameters to be set later
        self.train = train
        N = train['y'].size()[-1]
        self.v = [i for i in range(N)]
        self.epochs = int(np.ceil(UPDATES * MBSIZE / N))

        # save parameters from inputs
        self.model = model
        self.outcome = outcome
        self.isRecurrent = (model != 'arrival') or (outcome == 'days')
        self.isLSTM = outcome in ['days', 'delay']
        self.EM = outcome in ['sec', 'con']

        # parameters and loss function
        if self.EM:
            sizes['out'] *= params.K
            self.loss = BetaMixtureLoss
            vals = np.full(tuple(train['y'].size()) + (params.K,), 1/params.K)
            self.omega = torch.as_tensor(vals, dtype=torch.float).detach()
        elif self.outcome in ['days', 'hist']:
            self.loss = NegativeBinomialLoss
        else:
            self.loss = LogitLoss

        # neural net(s)
        if self.isRecurrent:
            if self.isLSTM:
                self.net = LSTM(params, sizes)
            else:
                self.net = RNN(params, sizes)
        else:
            self.net = FeedForward(params, sizes)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data, train=True):
        # train / eval mode
        self.net.train(train)

        # prediction using net
        if self.isRecurrent:
            x_time = rnn.pack_padded_sequence(data['x_time'], data['turns'])
            theta = self.net(data['x_fixed'], x_time)
        else:
            theta = self.net(data['x_fixed'])

        # outcome
        if self.isRecurrent:
            mask = ~torch.isnan(data['y'])
            data['y'] = data['y'][mask]
            theta = theta[mask]

        # calculate loss
        if 'omega' in data:
            if self.isRecurrent:
                loss, data['omega'][mask] = self.loss(
                    theta, data['y'], data['omega'][mask])
            else:
                loss, data['omega'] = self.loss(theta, data['y'], data['omega'])
            return loss, data['omega']
        else: 
            return self.loss(theta, data['y'])  


    def run_epoch(self):
        # batch indices
        np.random.shuffle(self.v)
        indices = np.array_split(self.v, 1 + len(self.v) // MBSIZE)

        # loop over batches
        lnL = 0
        for i in range(len(indices)):
            # zero gradient
            self.optimizer.zero_grad()

            # index data
            idx = torch.tensor(np.sort(indices[i]))
            data = {}
            data['x_fixed'] = torch.index_select(self.train['x_fixed'], -2, idx)
            data['y'] = torch.index_select(self.train['y'], -1, idx)
            if self.isRecurrent:
                data['x_time'] = self.train['x_time'][:,idx,:]
                data['turns'] = self.train['turns'][idx]
            if self.EM:
                data['omega'] = torch.index_select(self.omega, -2, idx)

            # calculate loss
            loss, omega = self.evaluate_loss(data)

            # update omega
            if self.EM:
                if self.isRecurrent:
                    self.omega[:, idx, :] = omega
                else:
                    self.omega[idx, :] = omega

            # step down gradients
            loss.backward()
            self.optimizer.step()

            # return log-likelihood
            lnL += -loss.item()

        return lnL
