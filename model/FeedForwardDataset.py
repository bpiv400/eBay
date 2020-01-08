import os, h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
from compress_pickle import load
from constants import INPUT_DIR, PARTS_DIR


class FeedForwardDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a dataset that extends torch.utils.data.Dataset
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # listing-level features
        self.x = h5py.File(PARTS_DIR + '{}/x_lstg.hdf5'.format(part), 'r')

        # number of labels
        self.N_labels = np.shape(self.d['y'])[0]

        # create single group for sampling
        self.groups = [np.array(range(self.N_labels))]


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # y is indexed directly
        y = self.d['y'][idx]

        # initialize x from listing-level features
        idx_x = self.d['idx_x'][idx]
        x = {k: v[idx_x,:] for k, v in self.x.items()}

        # append thread-level features
        if 'x_thread' in self.d:
            x['lstg'] = np.concatenate(
                (x['lstg'], self.d['x_thread'][idx]), axis=1)

        # append offer-level features
        if 'x_offer' in self.d:
            x = x.update(self.d['x_offer'][idx])

        return y, x
        

    def __len__(self):
        return self.N_labels