import sys
import torch
from torch.nn.utils import rnn
from datetime import datetime as dt
import numpy as np
from simulator.nets import *
from constants import *


# constructs model-specific neural network.
class Simulator:

    def __init__(self, model, sizes):
        '''
        model: one of 'arrival', 'hist', 'delay_byr', 'delay_slr',
            'con_byr', 'con_slr'
        sizes: dictionary of data sizes
        '''

        # save parameters from inputs
        self.model = model

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
            self.net = LSTM(sizes)
        else:
            self.net = FeedForward(sizes)    


    def evaluate_loss(self, d):
        # feed-forward
        if 'x_time' not in d:
            y = d['y'].to(DEVICE)
            theta = self.net(d['x']).squeeze()

            # for con_byr, split by turn and calculate loss separately
            if self.model == 'con_byr':
                # observation is on buyer's 4th turn if all three turn indicators are 0
                t4 = torch.sum(d['x']['lstg'][:,-3:], dim=1) == 0

                # loss for first 3 turns
                loss = self.loss[0](theta[~t4,:], y[~t4].long())

                # loss for last turn: use accept probability only
                if torch.sum(t4).item() > 0:
                    loss += self.loss[1](theta[t4,-1], (y[t4] == 100).float())

                return loss
            else:
                if self.loss.__str__() == 'BCEWithLogitsLoss()':
                    return self.loss(theta, y.float())
                else:
                    return self.loss(theta, y.long())

        # use mask for recurrent
        mask = d['y'] > -1

        # prediction from recurrent net
        theta = self.net(d['x'], d['x_time'])

        # apply mask and calculate loss
        theta = theta[mask]
        y = d['y'][mask].to(DEVICE)
        return self.loss(theta, y)


    def run_batch(self, d, optimizer=None):
        # train / eval mode
        isTraining = optimizer is not None
        self.net.train(isTraining)

        # zero gradient
        if isTraining:
            optimizer.zero_grad()

        # pack x_time
        if 'x_time' in d:
            d['x_time'] = rnn.pack_padded_sequence(
                d['x_time'], d['turns'], batch_first=True)

        # calculate loss
        loss = self.evaluate_loss(d)

        # step down gradients
        if isTraining:
            loss.backward()
            optimizer.step()

        # return log-likelihood
        return -loss.item()

