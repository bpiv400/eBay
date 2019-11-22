import sys
import torch
from torch.nn.utils import rnn
from datetime import datetime as dt
import numpy as np
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
        self.c = params['c']
        self.mbsize = params['mbsize']

        # sloss function
        if model in ['hist', 'con_slr']:
            self.loss = torch.nn.CrossEntropyLoss(
                reduction='sum')
        elif model == 'con_byr':
            self.loss = [torch.nn.CrossEntropyLoss(reduction='sum'),
                         torch.nn.BCEWithLogitsLoss(reduction='sum')]
        elif model == 'arrival':
            self.loss = torch.nn.PoissonNLLLoss(
                log_input=True, reduction='sum')
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(
                reduction='sum')

        # neural net(s)
        if ('delay' in model) or (model == 'arrival'):
            self.net = LSTM(params, sizes).to(self.device)
        else:
            self.net = FeedForward(params, sizes).to(self.device)     


    def evaluate_loss(self, d):
        # feed-forward
        if 'x_time' not in d:
            y = d['y']
            theta = self.net(d['x_fixed']).squeeze()

            # for con_byr, split by turn and calculate loss separately
            if self.model == 'con_byr':
                # observation is on buyer's 4th turn if all three turn indicators are 0
                t4 = np.sum(d['x_fixed'][:,-3:], axis=0) == 0


                theta0 = theta[0][mask[:,:3]]
                y0 = d['y'][:,:3][mask[:,:3]]
                loss0 = self.loss[0](theta0, y0)

                theta1 = theta[1][mask[:,3]]
                y1 = d['y'][:,3][mask[:,3]]
                loss1 = self.loss[1](theta1, y1)

                return loss0 + loss1
            else:
                
                return self.loss(theta, y)

        # use mask for recurrent
        mask = d['y'] > -1

        # prediction from recurrent net
        theta = self.net(d['x_fixed'], d['x_time'])

        # apply mask and calculate loss
        theta = theta[mask]
        y = d['y'][mask]
        return self.loss(theta, y)

        


    def apply_max_norm_constraint(self, eps=1e-8):
        for name, param in self.net.named_parameters():
            if 'bias' not in name and 'output' not in name:
                norm = param.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, 0, self.c)
                param = param * desired / (eps + norm)


    def run_batch(self, d, optimizer, isTraining):
        # train / eval mode
        self.net.train(isTraining)

        # zero gradient
        if isTraining:
            optimizer.zero_grad()

        # move to gpu
        if self.device != 'cpu':
            d = {k: v.to(self.device) for k, v in d.items()}

        # pack x_time
        if 'x_time' in d:
            d['x_time'] = rnn.pack_padded_sequence(
                d['x_time'], d['turns'], batch_first=True)

        # calculate loss
        loss = self.evaluate_loss(d)
        del d

        # step down gradients
        if isTraining:
            loss.backward()
            optimizer.step()
            #self.apply_max_norm_constraint()

        # return log-likelihood
        return -loss.item()

