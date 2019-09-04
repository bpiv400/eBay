import sys
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
    def __init__(self, model, outcome, train, params):
        # initialize parameters to be set later
        self.train = train
        self.N = torch.sum(~torch.isnan(train['y'])).item()
        self.v = [i for i in range(train['y'].size()[-1])]
        self.batches = int(np.ceil(len(self.v) / MBSIZE))

        # save parameters from inputs
        self.model = model
        self.outcome = outcome
        self.isRNN = model != 'arrival'
        self.EM = self.outcome in ['sec', 'con']

        # parameters and loss function
        if self.EM:
            self.K = params.K
            N_out = 3 * self.K
            self.loss = BetaMixtureLoss
            vals = np.full(tuple(train['y'].size()) + (self.K,), 1/self.K)
            self.omega = torch.as_tensor(vals, dtype=torch.float).detach()
        elif self.outcome in ['days', 'hist']:
            N_out = 2
            self.loss = NegativeBinomialLoss
        else:
            N_out = 1
            self.loss = LogitLoss

        # neural net(s)
        N_fixed = train['x_fixed'].size()[-1]
        if self.isRNN:
            self.steps = train['y'].size()[0]
            N_time = train['x_time'].size()[-1]
            self.net = RNN(N_fixed, N_time, N_out, params.hidden)
        else:
            self.net = FeedForward(N_fixed, N_out, params.hidden)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)


    def run_epoch(self):
        # batch indices
        np.random.shuffle(self.v)
        indices = np.array_split(self.v, 1 + len(self.v) // MBSIZE)

        # loop over batches
        lnL = 0
        for i in range(self.batches):
            idx = torch.tensor(np.sort(indices[i]))

            # zero gradient
            self.optimizer.zero_grad()

            # prediction using net
            x_fixed = torch.index_select(self.train['x_fixed'], -2, idx)
            if self.isRNN:
                x_time = rnn.pack_padded_sequence(
                    self.train['x_time'][:, idx, :],
                    self.train['turns'][idx])
                theta = self.net(x_fixed, x_time, self.steps)
            else:
                theta = self.net(x_fixed)

            # outcome
            y = torch.index_select(self.train['y'], -1, idx)
            if self.isRNN:
                mask = ~torch.isnan(y)
                y = y[mask]
                theta = theta[mask]

            # calculate loss
            if self.EM:
                omega = torch.index_select(self.omega, -2, idx)
                if self.isRNN:
                    loss, omega[mask] = self.loss(theta, omega[mask], y)
                    self.omega[:, idx, :] = omega
                else:
                    loss, omega = self.loss(theta, omega, y)
                    self.omega[idx, :] = omega
            else:
                loss = self.loss(theta, y)

            # step down gradients
            loss.backward()
            self.optimizer.step()

            # return log-likelihood
            lnL += -loss.item()

        return lnL / self.N
