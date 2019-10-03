import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
import h5py, pickle

sys.path.append('repo/')
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, partition, model, outcome):

        # save parameters to self
        self.partition = partition
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
            return y, x_fixed
        
        # role models are recurrent
        x_time = self.d['x_time'][idx,:]
        turns = self.d['turns'][idx]

        return y, x_fixed, x_time, turns

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
        self.batches = np.array_split(v, N+1 // MBSIZE)
        print('Batch count: %d' % len(self.batches))

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
    y, x_fixed = batch
    return torch.from_numpy(y), torch.from_numpy(x_fixed)

