import sys
import numpy as np, pandas as pd
import torch
from datetime import datetime as dt
from torch.utils.data import Dataset, Sampler
from compress_pickle import load
from constants import *


# defines a sampler that extends torch.utils.data.Sampler
class Sample(Sampler):

    def __init__(self, data, mbsize, isTraining):
        """
        data: instance of Inputs
        mbsize: scalar minibatch size
        isTraining: chop into minibatches if True
        """
        super().__init__(None)

        # chop into minibatches; shuffle for training
        self.batches = []
        for v in data.d['groups']:
            if isTraining:
                np.random.shuffle(v)
            self.batches += np.array_split(v, 1 + len(v) // mbsize)
        # shuffle training batches
        if isTraining:
            np.random.shuffle(self.batches)


    def __iter__(self):
        """
        Iterate over batches defined in intialization
        """
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# collate function for feedforward networks
def collateFF(batch):
    y, x_fixed, idx = [], [], []
    for b in batch:
        y.append(b[0])
        x_fixed.append(torch.from_numpy(b[1]))
        idx.append(b[2])

    # convert to tensor
    y = torch.from_numpy(np.asarray(y))
    x_fixed = torch.stack(x_fixed).float()
    idx = torch.tensor(idx)

    # output is dictionary of tensors
    return {'y': y, 'x_fixed': x_fixed}


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, turns, x_fixed, x_time, idx = [], [], [], [], []

    # sorts the batch list in decreasing order of turns
    for b in batch:
        y.append(torch.from_numpy(b[0]))
        turns.append(b[1])
        x_fixed.append(torch.from_numpy(b[2]))
        x_time.append(torch.from_numpy(b[3]))
        idx.append(b[4])

    # convert to tensor, pack if needed
    y = torch.stack(y)
    turns = torch.from_numpy(np.asarray(
        turns, dtype='int64')).long()
    x_fixed = torch.stack(x_fixed).float()
    x_time = torch.stack(x_time, dim=0).float()
    idx = torch.tensor(idx)
    
    # output is dictionary of tensors
    return {'y': y, 
            'turns': turns,
            'x_fixed': x_fixed, 
            'x_time': x_time}
