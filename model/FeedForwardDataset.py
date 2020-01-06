import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
from compress_pickle import load
from constants import *


class FeedForwardDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a dataset that extends torch.utils.data.Dataset
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

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
        # y and x are indexed directly
        y = self.d['y'][idx]
        x = {k: v[idx,:].astype(np.float32) for k, v in self.d['x'].items()}

        return y, x
        

    def __len__(self):
        return self.N_labels