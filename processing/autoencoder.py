import torch, torch.nn as nn
from constants import *


class Autoencoder(nn.Module):
    def __init__(self, N, layers, hidden1, hidden2):
        # super constructor
        super(Autoencoder, self).__init__()

        # activation function
        f = nn.ReLU()

        # initial layer
        self.seq = nn.ModuleList([nn.Linear(N, hidden1), f])

        # intermediate layers
        for i in range(layers):
            self.seq.append(nn.Linear(hidden1, hidden2))
            self.seq.append(f)

        # output layer
        self.seq.append([nn.Linear(hidden1, hidden2), f, nn.Linear(hidden2, N)])

    def forward(self, x):
        for _, m in enumerate(self.seq):
            x = m(x)
        return x