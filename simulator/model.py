import sys, math
import torch, torch.optim as optim
from torch.nn.utils import rnn
from datetime import datetime as dt
import numpy as np
from simulator.loss import *
from simulator.nets import *
from constants import *


# constructs model-specific neural network.
class Simulator:

    def __init__(self, model, params, sizes, device='cpu'):
        '''
        model: one of 'arrival', 'hist', 'delay_byr', 'delay_slr',
            'con_byr', 'con_slr'
        params: dictionary of neural net parameters
        sizes: dictionary of data sizes
        device: either 'cpu' or 'gpu'
        '''

        # save parameters from inputs
        self.model = model
        self.device = device
        self.isRecurrent = model != 'hist'
        self.c = params['c']
        self.mbsize = params['mbsize']

        # size of theta and loss function
        if model in ['hist', 'con_byr', 'con_slr']:
            self.loss = cross_entropy_loss
        else:
            if model == 'arrival':
                self.loss = poisson_loss
            else:
                self.loss = logit_loss

        # neural net(s)
        if not self.isRecurrent:
            self.net = FeedForward(params, sizes).to(self.device)
        elif ('delay' in model) or (model == 'arrival'):
            self.net = LSTM(params, sizes).to(self.device)
        else:
            self.net = RNN(params, sizes).to(self.device)          

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), 
            betas=(0.9, 1-math.pow(10, params['b2'])),
            lr=math.pow(10, params['lr']))

    def evaluate_loss(self, d):
        # feed-forward
        if not self.isRecurrent:
            y = d['y']
            theta = self.net(d['x_fixed']).squeeze()

        # use mask for recurrent
        else:
            # retain indices with outcomes
            mask = d['y'] > -1

            # prediction from recurrent net
            theta = self.net(d['x_fixed'], d['x_time'])

            # separate last turn of con_byr model
            if self.model == 'con_byr':
                # split y by turn
                y = [d['y'][:,:3], d['y'][:,3]]

                # apply mask separately
                theta = [theta[0][mask[:,:3]], theta[1][mask[:,3]]]
                y = [y[0][mask[:,:3]], y[1][mask[:,3]]]
            else:
                theta = theta[mask]
                y = d['y'][mask]
            
        # calculate loss
        return self.loss(theta, y)

    def apply_max_norm_constraint(self, eps=1e-8):
        for name, param in self.net.named_parameters():
            if 'bias' not in name and 'output' not in name:
                norm = param.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, 0, self.c)
                param = param * desired / (eps + norm)


    def run_batch(self, d, isTraining):
        # train / eval mode
        self.net.train(isTraining)

        # zero gradient
        if isTraining:
            self.optimizer.zero_grad()

        # move to gpu
        if self.device != 'cpu':
            d = {k: v.to(self.device) for k, v in d.items()}

        if self.isRecurrent:
            d['x_time'] = rnn.pack_padded_sequence(
                d['x_time'], d['turns'], batch_first=True)

        # calculate loss
        loss = self.evaluate_loss(d)
        del d

        # step down gradients
        if isTraining:
            loss.backward()
            self.optimizer.step()
            self.apply_max_norm_constraint()

        # return log-likelihood
        return -loss.item()

