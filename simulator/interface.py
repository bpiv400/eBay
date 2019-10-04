import sys
import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset, Sampler
import h5py, pickle

sys.path.append('repo/')
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, partition, model, outcome):

        # save parameters to self
        self.model = model
        self.outcome = outcome

        # path
        path = 'data/inputs/%s/%s_%s.hdf5' % (partition, model, outcome)

        # load data file
        self.d = h5py.File(path, 'r')

        # save length
        self.N = np.shape(self.d['x_fixed'])[0]


    def close(self):
        self.d.close()


    def __getitem__(self, idx):
        # all models have y and x_fixed
        y = self.d['y'][idx]
        x_fixed = self.d['x_fixed'][idx,:]

        # arrival models are feed-forward
        if self.model == 'arrival':
            return y, x_fixed, idx
        
        # role models are recurrent
        x_time = self.d['x_time'][:,idx,:]
        turns = self.d['turns'][idx]

        return y, x_fixed, x_time, turns, idx


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
    y = torch.tensor(y)
    x_fixed = torch.stack(x_fixed)
    idx = torch.tensor(idx)

    # output is (dictionary, indices)
    return {'y': y, 'x_fixed': x_fixed}, idx


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, x_fixed, x_time, turns, idx = [], [], [], [], []

    # sorts the batch list in decreasing order of turns
    ordered = sorted(batch, key=lambda x: len(x[3]), reverse=True)
    for b in ordered:
        y.append(b[0])
        x_fixed.append(torch.from_numpy(b[1]))
        x_time.append(torch.from_numpy(b[2]))
        turns.append(b[3])
        idx.append(b[4])

    # convert to tensor, pack if needed
    y = torch.tensor(y).float()
    x_fixed = torch.stack(x_fixed)
    x_time = rnn.pack_padded_sequence(
        torch.stack(x_time, dim=1), torch.from_numpy(turns))
    idx = torch.tensor(idx)

    # output is (dictionary, indices)
    return {'y': y, 'x_fixed': x_fixed, 'x_time': x_time}, idx
