import sys
import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset, Sampler
from compress_pickle import load
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, part, model):

        # save parameters to self
        self.model = model

        # inputs
        self.d = load('%s/inputs/%s/%s.gz' % (PREFIX, part, model))

        # save length
        self.N = np.shape(self.d['x_fixed'])[0]


    def __getitem__(self, idx):
        # all models index y and x_fixed using idx
        y = self.d['y'][idx]
        x_fixed = self.d['x_fixed'][idx,:]

        # feed-forward models
        if self.model == 'hist':
            return y, x_fixed, idx

        # x_time
        if self.model == 'arrival':
            # number of hours
            n = MAX_DAYS * 24

            # index of first hour
            idx_hour = self.d['idx_hour'][idx] + \
                np.array(range(n), dtype='uint16')

            # hour features
            x_hour = self.d['x_hour'][idx_hour].astype('float32')

            # time feats
            x_time = x_hour

        else:
            x_time = self.d['x_time'][idx,:,:]
        
        return y, x_fixed, x_time, idx


    def __len__(self):
        return self.N


# defines a sampler that extends torch.utils.data.Sampler
class Sample(Sampler):

    def __init__(self, dataset):
        """
        dataset: instance of Inputs
        """
        super().__init__(None)

        # save dataset to self
        self.dataset = dataset
        
        # shuffle indices and batch
        N = len(dataset)
        v = [i for i in range(N)]
        np.random.shuffle(v)
        self.batches = np.array_split(v, 1 + N // MBSIZE)

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

    # output is (dictionary, indices)
    return {'y': y, 'x_fixed': x_fixed}, idx


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, x_fixed, x_time, idx = [], [], [], []

    # sorts the batch list in decreasing order of turns
    for b in batch:
        y.append(b[0])
        x_fixed.append(torch.from_numpy(b[1]))
        x_time.append(torch.from_numpy(b[2]))
        idx.append(b[3])

    # convert to tensor, pack if needed
    y = torch.from_numpy(np.asarray(y))
    turns = torch.sum(y > -1, dim=1)
    x_fixed = torch.stack(x_fixed).float()
    x_time = torch.stack(x_time, dim=0).float()
    idx = torch.tensor(idx)

    # output is (dictionary, indices)
    return {'y': y, 
            'x_fixed': x_fixed, 
            'x_time': x_time, 
            'turns': turns}, idx
