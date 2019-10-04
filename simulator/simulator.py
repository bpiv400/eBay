import sys, math
sys.path.append('../')
from constants import *
import torch, torch.optim as optim
import numpy as np
from loss import *
from nets import *


class Simulator:
    '''
    Constructs neural network and holds omega for beta mixture model.
    '''
    def __init__(self, model, outcome, params, sizes):
        # save parameters from inputs
        self.model = model
        self.outcome = outcome
        self.isRecurrent = model != 'arrival'
        self.isLSTM = outcome == 'delay'
        self.EM = outcome in ['sec', 'con']

        # parameters and loss function
        if self.EM:
            sizes['out'] *= params.K
            self.loss = BetaMixtureLoss
            vals = np.full(tuple(sizes['N'], 1) + (params.K,), 1/params.K)
            self.omega = torch.as_tensor(vals, dtype=torch.float).detach()
        elif self.outcome in ['days', 'hist']:
            self.loss = NegativeBinomialLoss
        else:
            self.loss = LogitLoss

        # neural net(s)
        if self.isRecurrent:
            if self.isLSTM:
                self.net = LSTM(params, sizes).to(DEVICE)
            else:
                self.net = RNN(params, sizes).to(DEVICE)
        else:
            self.net = FeedForward(params, sizes).to(DEVICE)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data, train=True):
        # train / eval mode
        self.net.train(train)

        # prediction using net
        if self.isRecurrent:
            theta = self.net(data['x_fixed'], data['x_time'])
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
                loss, data['omega'] = self.loss(
                    theta, data['y'], data['omega'])
            return loss, data['omega']
        else: 
            return self.loss(theta, data['y'])  


    def run_batch(self, data, idx):
        # zero gradient
        self.optimizer.zero_grad()

        # include omega for expectation-maximization
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
        return -loss.item()

