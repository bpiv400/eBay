import sys, math
import torch, torch.optim as optim
import numpy as np

sys.path.append('repo/')
from constants import *

sys.path.append('repo/simulator/')
from loss import *
from nets import *


class Simulator:
    '''
    Constructs neural network and holds omega for beta mixture model.
    '''
    def __init__(self, model, params, sizes, device='cpu'):
        # save parameters from inputs
        self.isRecurrent = model != 'arrival'

        # size of theta and loss function
        if 'con' in model:
            sizes['out'] = 2 + CON_SEGMENTS
            self.loss = emd_loss
        else:
            sizes['out'] = 1
            if model == 'arrival':
                self.loss = poisson_loss
            else:
                self.loss = logit_loss

        # neural net(s)
        if self.isRecurrent:
            if 'delay' in model:
                self.net = LSTM(params, sizes).to(device)
            else:
                self.net = RNN(params, sizes).to(device)
        else:
            self.net = FeedForward(params, sizes).to(device)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data, train=True):
        # train / eval mode
        self.net.train(train)

        # prediction using net
        if self.isRecurrent:
            theta = self.net(data['x_fixed'], data['x_time'])
            mask = data['y'] > -1
            data['y'] = data['y'][mask]
            theta = theta[mask,:]
        else:
            theta = self.net(data['x_fixed'])

        # calculate loss
        return self.loss(theta.squeeze(), data['y'])  


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

