import sys
import torch, torch.optim as optim
from torch.nn.utils import rnn
from datetime import datetime as dt
import numpy as np
from loss import *
from nets import *
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
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data):
        # feed-forward
        if not self.isRecurrent:
            y = data['y']
            theta = self.net(data['x_fixed']).squeeze()

        # use mask for recurrent
        else:
            # retain indices with outcomes
            mask = data['y'] > -1

            # prediction from recurrent net
            theta = self.net(data['x_fixed'], data['x_time'])

            # separate last turn of con_byr model
            if self.model == 'con_byr':
                # split y by turn
                y = [data['y'][:,:3], data['y'][:,3]]

                # apply mask separately
                theta = [theta[0][mask[:,:3]], theta[1][mask[:,3]]]
                y = [y[0][mask[:,:3]], y[1][mask[:,3]]]
            else:
                theta = theta[mask]
                y = data['y'][mask]
            
        # calculate loss
        return self.loss(theta, y)


    def run_batch(self, data, isTraining):
        # train / eval mode
        self.net.train(isTraining)

        # zero gradient
        if isTraining:
            self.optimizer.zero_grad()

        # move to gpu
        if self.device != 'cpu':
            data = {k: v.to(self.device) for k, v in data.items()}

        if self.isRecurrent:
            data['x_time'] = rnn.pack_padded_sequence(
                data['x_time'], data['turns'], batch_first=True)

        # calculate loss
        loss = self.evaluate_loss(data)

        # step down gradients
        if isTraining:
            loss.backward()
            self.optimizer.step()

        # return log-likelihood
        return -loss.item()

