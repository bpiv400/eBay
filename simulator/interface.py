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
        self.path = 'data/inputs/%s/%s_%s.hdf5' % (partition, model, outcome)

        # save length
        d = h5py.File(self.path, 'r')
        self.N = np.shape(d['x_fixed'])[0]
        d.close()


    def __getitem__(self, idx):
        # load data file
        with h5py.File(self.path, 'r') as d:
            # all models have y and x_fixed
            y = d['y'][idx]
            x_fixed = d['x_fixed'][idx,:]

            # arrival models are feed-forward
            if self.model == 'arrival':
                return y, x_fixed, idx
            
            # role models are recurrent
            x_time = d['x_time'][idx,:,:]

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
    y = torch.from_numpy(np.asarray(y)).float()
    x_fixed = torch.stack(x_fixed).float()
    idx = torch.tensor(idx)

    # output is (dictionary, indices)
    return {'y': y, 'x_fixed': x_fixed}, idx


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, x_fixed, x_time, idx = [], [], [], []

    # sorts the batch list in decreasing order of turns
    ordered = sorted(batch, key=lambda x: np.sum(np.isnan(x[0])))
    for b in ordered:
        y.append(b[0])
        x_fixed.append(torch.from_numpy(b[1]))
        x_time.append(torch.from_numpy(b[2]))
        idx.append(b[3])

    # convert to tensor, pack if needed
    y = torch.from_numpy(np.asarray(y)).float()
    turns = torch.sum(~torch.isnan(y), dim=1)
    x_fixed = torch.stack(x_fixed).float()
    x_time = torch.stack(x_time, dim=0).float()
    x_time = rnn.pack_padded_sequence(x_time, turns, batch_first=True)
    idx = torch.tensor(idx)

    # output is (dictionary, indices)
    return {'y': y, 'x_fixed': x_fixed, 'x_time': x_time}, idx
