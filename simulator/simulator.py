import sys
import torch, torch.optim as optim
import numpy as np
from constants import *
from simulator.loss import *
from simulator.nets import *


class Simulator:
    '''
    Constructs neural network and holds omega for beta mixture model.
    '''
    def __init__(self, model, params, sizes, device='cpu'):
        # save parameters from inputs
        self.model = model

        # size of theta and loss function
        if model in ['hist', 'con_byr', 'con_slr']:
            self.loss = emd_loss
            if model == 'hist':
                sizes['out'] = 100
            else:
                sizes['out'] = 2 + CON_SEGMENTS
        else:
            sizes['out'] = 1
            if model == 'arrival':
                self.loss = poisson_loss
            else:
                self.loss = logit_loss

        # neural net(s)
        if 'delay' in model:
            self.net = LSTM(params, sizes).to(device)
        elif 'con' in model:
            self.net = RNN(params, sizes).to(device)
        else:
            self.net = FeedForward(params, sizes).to(device)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data, train=True):
        # train / eval mode
        self.net.train(train)

        # prediction using net
        if model in ['arrival', 'hist']:
            theta = self.net(data['x_fixed'])
        else:
            mask = data['y'] > -1
            data['y'] = data['y'][mask]

            if model == 'con_byr':
                t, t4 = self.net(data['x_fixed'], data['x_time'])
                theta = [t[mask], t4[mask]]
            else:
                theta = self.net(data['x_fixed'], data['x_time'])
                theta = theta[mask]

        # calculate loss
        return self.loss(theta, data['y'])  


    def run_batch(self, data, idx):
        # zero gradient
        self.optimizer.zero_grad()

        # calculate loss
        loss = self.evaluate_loss(data)

        # step down gradients
        loss.backward()
        self.optimizer.step()

        # return log-likelihood
        return -loss.item()

