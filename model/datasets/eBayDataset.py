import os, h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from torch.utils.data import Dataset
from compress_pickle import load
from constants import INPUT_DIR, HDF5_DIR


class eBayDataset(Dataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a parent class that extends torch.utils.data.Dataset.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # number of labels
        self.N = len(self.d['y'])

        # listings file
        self.x = None
        if part == 'small':
            self.x_path = HDF5_DIR + 'train_models/x_lstg.hdf5'
        else:
            self.x_path = HDF5_DIR + '{}/x_lstg.hdf5'.format(part)
        
        # offer groups for embedding
        self.offer_keys = [k for k in sizes['x'] if k.startswith('offer')]


    def _init_subprocess(self):
        '''
        To be called in in first __getitem__ call in each subprocess. Loads HDF5 files.
        '''
        self.x = h5py.File(self.x_path, 'r')


    def _construct_x(self, idx):
        '''
        Returns a dictionary of grouped input features.
        :param idx: index of example.
        :return: dictionary of grouped input features at index idx.
        '''
        # initialize x from listing-level features
        idx_x = self.d['idx_x'][idx]
        x = {k: v[idx_x, :] for k, v in self.x.items()}

        # append thread-level features
        if 'x_thread' in self.d:
            x['lstg'] = np.concatenate(
                (x['lstg'], self.d['x_thread'][idx]))

        # append offer-level features
        for k in self.offer_keys:
            x[k] = self.d[k][idx]

        return x


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # initialize subprocess with hdf5 files
        if self.x is None:
            self._init_subprocess()

        # y is indexed directly
        y = self.d['y'][idx]

        # initialize x from listing-level features
        x = self._construct_x(idx)

        return y, x
        

    def __len__(self):
        return self.N