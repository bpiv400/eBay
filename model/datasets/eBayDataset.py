import os, h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from torch.utils.data import Dataset
from compress_pickle import load
from constants import INPUT_DIR, HDF5_DIR


class eBayDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a parent class that extends torch.utils.data.Dataset.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        # dictionary of inputs
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # path to listing-level features
        self.x = None
        self.path = HDF5_DIR + '{}/x_lstg.hdf5'.format(part)

        # parameters to be created in children
        self.N_labels = None
        self.groups = None


    def _construct_x(self, idx):
        '''
        Returns a dictionary of grouped input features.
        :param idx: index of example.
        :return: dictionary of grouped input features at index idx.
        '''
        # initialize hdf5 file on each subprocess
        if self.x is None:
            self.x = h5py.File(self.path, 'r')

        # initialize x from listing-level features
        idx_x = self.d['idx_x'][idx]
        x = {k: v[idx_x, :] for k, v in self.x.items()}

        # append thread-level features
        if 'x_thread' in self.d:
            l = np.array([a[idx] for a in self.d['x_thread']], 
                dtype='float32')
            x['lstg'] = np.concatenate((x['lstg'], l))

        # append offer-level features
        if 'x_offer' in self.d:
            for k, v in self.d['x_offer'].items():
                x[k] = np.array([a[idx] for a in v], dtype='float32')

        return x


    def __getitem__(self, idx):
        return NotImplementedError()
        

    def __len__(self):
        return self.N_labels