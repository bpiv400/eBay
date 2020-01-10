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
        # paths to hdf5 files
        self.x_path = HDF5_DIR + '{}/x_lstg.hdf5'.format(part)
        self.d_path = HDF5_DIR + '{}/{}.hdf5'.format(part, name)

        # number of labels
        self.N_labels = sizes['N_labels']

        # offer groups for embedding
        self.offer_keys = [k for k in sizes['x'] if k.startswith('offer')]

        # to be loaded in subprocess
        self.x = None
        self.d = None

        # groups for sampling to be created in children
        self.groups = None


    def _init_subprocess(self):
        '''
        To be called in in first __getitem__ call in each subprocess. Loads HDF5 files.
        '''
        self.x = h5py.File(self.x_path, 'r')
        self.d = h5py.File(self.d_path, 'r')


    def _fill_array(self, a, stub, idx):
        '''
        Creates a semi-sparse numpy array by occasionally replacing zeros with values.
        :param a: numpy array of zeros.
        :param stub: string such that d has keys [stub, stub + '_periods', 'idx_' + stub]
        :param idx: int observation index.
        :return: semi-sparse numpy array.
        '''
        idx_array = self.d['idx_' + stub][idx]
        if idx_array > -1:
            l = self.d[stub + '_periods'][idx].decode("utf-8").split('/')
            for p in l:
                assert p != ''
                a[int(p)] = self.d[stub][idx_array]
                idx_array += 1
        return a


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
        return NotImplementedError()
        

    def __len__(self):
        return self.N_labels