import sys
import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from datetime import datetime as dt
from torch.utils.data import Dataset
from compress_pickle import load
from constants import *


class eBayDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a dataset that extends torch.utils.data.Dataset
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load('{}/inputs/{}/{}.gz'.format(PREFIX, part, name))

        # number of examples and labels
        self.N_examples = np.shape(self.d['y'])[0]
        self.N_labels = np.sum(self.d['y'] > -1)

        # create single group if groups do not exist
        if 'groups' not in self.d:
            self.d['groups'] = [np.array(range(self.N_examples))]

        # recurrent or feed-forward
        self.isRecurrent = len(np.shape(self.d['y'])) > 1

        # collate function
        self.collate = collateRNN if self.isRecurrent else collateFF


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        raise NotImplementedError()
        

    def __len__(self):
        return self.N_examples


class ModelDataset(eBayDataset):
    def __init__(self, part, name):
        super(ModelDataset, self).__init__(part, name)

        # for arrival and delay models
        if 'tf' in self.d:
            # number of time steps
            self.n = np.shape(self.d['y'])[1]

            # counter for expanding clock features
            role = name.split('_')[-1]
            interval = int(INTERVAL[role] / 60) # interval in minutes
            self.counter = interval * np.array(range(self.n), dtype='uint16')

            # period / max periods
            self.duration = np.expand_dims(
                np.array(range(self.n), dtype='float32') / self.n, axis=1)

            # number of time features
            for val in self.d['tf'].values():
                N_tfeats = len(val.columns)
                break
                
            # empty time feats
            self.tf0 = np.zeros((self.n, N_tfeats), dtype='float32')


    def __getitem__(self, idx):
        # y and x are indexed directly
        y = self.d['y'][idx]
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        # feed-forward models
        if not self.isRecurrent:
            return y, x

        # indices of timestamps
        idx_clock = self.d['idx_clock'][idx] + self.counter

        # clock features
        x_clock = self.d['x_clock'][idx_clock].astype('float32')

        # fill in missing time feats with zeros
        if idx in self.d['tf']:
            x_tf = self.d['tf'][idx].reindex(
                index=range(self.n), fill_value=0).to_numpy()
        else:
            x_tf = self.tf0

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate((x_clock, x_tf, self.duration), axis=1)

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, x, x_time


def DiscrimDataset(eBayDataset):
    def __init__(self, part, name):
        super(ModelDataset, self).__init__(part, name)


    def __getitem__(self, idx):
        # y is indexed directly
        y = self.d['y'][idx]

        # components of x are indexed using idx_x
        idx_x = self.d['idx_x'][idx]
        x = {k: v[idx_x] for k, v in x.items()}

        return y, x


def collateFF(batch):
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


def collateRNN(batch):
    '''
    Converts examples to tensors for a recurrent network.
    :param batch: list of (dictionary of) numpy arrays.
    :return: dictionary of (dictionary of) tensors.
    '''
    y, x, x_time = [], {}, []

    # sorts the batch list in decreasing order of turns
    for b in batch:
        y.append(torch.from_numpy(b[0]))
        for k, v in b[1].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]
        x_time.append(torch.from_numpy(b[2]))

    # convert to tensor
    y = torch.stack(y).float()
    turns = torch.sum(y > -1, dim=1).long()
    x = {k: torch.stack(v).float() for k, v in x.items()}
    x_time = torch.stack(x_time, dim=0).float()

    # slice off censored observations
    y = y[:,:turns[0]]
    x_time = x_time[:,:turns[0],:]

    # pack for recurrent network
    x_time = rnn.pack_padded_sequence(x_time, turns, batch_first=True)

    # output is dictionary of tensors
    return {'y': y, 'x': x, 'x_time': x_time}