import torch, torch.nn as nn
from torch.autograd import Function
from utils import *
from loss import *


class Simulator(nn.Module):
    def __init__(self, N_fixed, N_offer, model, params):
        # super constructor
        super(Simulator, self).__init__()

        # save parameters to self
        self.N_layers = params.layers
        self.model = model
        if model in ['delay', 'con']:
            self.M = 2 if model == 'con' else 1
            self.K = params.K
            N_out = self.M + 2 * params.K
            self.criterion = BetaMixtureLoss.apply
        else:
            self.M = N_out = 1
            self.K = 0
            self.criterion = BinaryLoss.apply

        # initial hidden nodes and LSTM cell
        self.h0 = nn.Linear(N_fixed, params.hidden)
        self.c0 = nn.Linear(N_fixed, params.hidden)

        # activation function
        if params.f == 'relu':
            self.f = nn.ReLU()
        elif params.f == 'sigmoid':
            self.f = nn.Sigmoid()

        # lstm layer
        self.lstm = nn.LSTM(input_size=N_offer, hidden_size=params.hidden,
                bias=True, num_layers=params.layers, dropout=params.dropout)

        # output layer
        self.output = nn.Linear(params.hidden, N_out)


    def get_model(self):
        return self.model


    def get_K(self):
        return self.K


    def get_M(self):
        return self.M


    def get_criterion(self):
        return self.criterion


    def forward(self, x_fixed, x_offer):
        # initialize model
        x_fixed = x_fixed.repeat(self.N_layers, 1, 1)
        init = (self.f(self.h0(x_fixed)), self.f(self.c0(x_fixed)))
        theta, _ = self.lstm(x_offer, init)

        # ensure that predictions are padded to MAX_TURNS
        theta, _ = nn.utils.rnn.pad_packed_sequence(theta, total_length=3)

        # exponential transform: (seq_len, batch_size, N_output)
        theta = torch.exp(self.output(theta))

        # return partition
        eta = theta[:,:,:self.M]
        p = torch.div(eta, 1 + torch.sum(eta, dim=2, keepdim=True))
        a = theta[:,:,self.M:self.K+self.M] + 1
        b = theta[:,:,self.K+self.M:] + 1
        return p, a, b