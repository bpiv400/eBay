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
        self.d = load('{}/inputs/{}/{}.gz'.format(PREFIX, part, name))

        # number of labels
        self.N_labels = np.shape(self.d['y'])[0]

        # create single group for sampling
        self.d['groups'] = [np.array(range(self.N_examples))]


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # y and x are indexed directly
        y = self.d['y'][idx]
        x = {k: v[idx,:] for k, v in self.d['x'].items()}

        return y, x
        

    def __len__(self):
        return self.N_labels


    def collate(self, batch):
        '''
        Converts examples to tensors for a feed-forward network.
        :param batch: list of (dictionary of) numpy arrays.
        :return: dictionary of (dictionary of) tensors.
        '''
        y, x = [], {}
        for b in batch:
            y.append(b[0])
            for k, v in b[1].items():
                if k in x:
                    x[k].append(torch.from_numpy(v))
                else:
                    x[k] = [torch.from_numpy(v)]

        # convert to (single) tensors
        y = torch.from_numpy(np.asarray(y)).long()
        x = {k: torch.stack(v).float() for k, v in x.items()}

        # output is dictionary of tensors
        return {'y': y, 'x': x}