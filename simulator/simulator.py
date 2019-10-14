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
    def __init__(self, model, outcome, params, sizes, device='cpu'):
        # save parameters from inputs
        self.model = model
        self.outcome = outcome
        self.isRecurrent = model != 'arrival'
        self.isLSTM = outcome == 'delay'
        self.EM = outcome in ['sec', 'con']
        self.device = device

        # parameters and loss function
        if self.EM:
            sizes['out'] = 2 * params['K']
            self.loss = beta_mixture_loss      
            
            # initialize omega to 1/K
            if self.isRecurrent:
                vals = np.full(
                    (sizes['N'], sizes['steps'],) + (params['K'],), 
                    1/params['K'])
            else:
                vals = np.full((sizes['N'],) + (params['K'],), 
                    1/params['K'])
            self.omega = torch.as_tensor(vals, dtype=torch.float).detach()
        else:
            sizes['out'] = 1
            if self.outcome == 'days':
                self.loss = poisson_loss
            else:
                self.loss = logit_loss

        # neural net(s)
        if self.isRecurrent:
            if self.isLSTM:
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
        else:
            theta = self.net(data['x_fixed'])

        # outcome
        if self.isRecurrent:
            mask = data['y'] > -1
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
            data['omega'] = self.omega[idx, :].to(self.device)

        # calculate loss
        loss, omega = self.evaluate_loss(data)

        # update omega
        if self.EM:
            self.omega[idx, :] = omega.to('cpu')

        # step down gradients
        loss.backward()
        self.optimizer.step()

        # return log-likelihood
        return -loss.item()

