import sys
import torch, torch.optim as optim
import numpy as np
from constants import *
from loss import *
from nets import *


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

        # size of theta and loss function
        if model in ['hist', 'con_byr', 'con_slr']:
            self.loss = emd_loss
            self.dim = torch.from_numpy(sizes['dim']).unsqueeze(
                dim=0).to(device)  # bucket values
            if model == 'hist':
                sizes['out'] = 100
            else:
                sizes['out'] = 1 + CON_SEGMENTS
        else:
            sizes['out'] = 1
            if model == 'arrival':
                self.loss = poisson_loss
            else:
                self.loss = logit_loss

        # neural net(s)
        if 'delay' in model:
            self.net = LSTM(params, sizes).to(self.device)
        elif 'con' in model:
            self.net = RNN(params, sizes).to(self.device)
        else:
            self.net = FeedForward(params, sizes).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def evaluate_loss(self, data, train=True):
        # train / eval mode
        self.net.train(train)

        # prediction using net
        if self.model in ['arrival', 'hist']:
            theta = self.net(data['x_fixed'])

            if self.model == 'hist':
                y = torch.pow(y.unsqueeze(dim=-1) - self.dim, 2)
        else:
            mask = data['y'] > -1
            

            # separate output for last turn of con_byr model
            if self.model == 'con_byr':
                t, t4 = self.net(data['x_fixed'], data['x_time'])
                theta = [t[mask[:,:3]], t4[mask[:,3]]]
                y = [data['y'][mask[:,:3]], data['y'][mask[:,3]]]
                y[0] = torch.pow(y.unsqueeze(dim=-1) - self.dim, 2)

            else:
                theta = self.net(data['x_fixed'], data['x_time'])
                theta = theta[mask]
                y = data['y'][mask]

                if self.model == 'con_slr':
                   y = torch.pow(y.unsqueeze(dim=-1) - self.dim, 2)

        # calculate loss
        return self.loss(theta, y)


    def run_batch(self, data, idx):
        # zero gradient
        self.optimizer.zero_grad()

        # move to gpu
        data = {k: v.to(self.device) for k, v in data.items()}
        idx = idx.to(self.device)

        # calculate loss
        loss = self.evaluate_loss(data)

        # step down gradients
        loss.backward()
        self.optimizer.step()

        # return log-likelihood
        return -loss.item()

