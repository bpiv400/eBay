import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset
from compress_pickle import load
from constants import *
from utils import extract_clock_feats


class RecurrentDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a dataset that extends torch.utils.data.Dataset
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load('{}/inputs/{}/{}.gz'.format(PREFIX, part, name))

        # add x_clock to dictionary
        self.d['x_clock'] = load('{}/inputs/universal/x_clock.gz'.format(PREFIX))

        # number of examples and labels
        self.N_examples = np.shape(self.d['y'])[0]
        self.N_labels = np.sum(self.d['y'] > -1)

        # number of time steps
        role = name.split('_')[-1]
        self.T = INTERVAL_COUNTS[role]

        # counter for expanding clock features
        self.counter = INTERVAL[role] * np.array(range(self.T))

        # period / max periods
        self.duration = np.expand_dims(
            np.array(range(self.T), dtype='float32') / self.T, axis=1)

        # empty time feats
        for val in self.d['tf'].values():
            N_tfeats = len(val.columns)
            break
        self.tf0 = np.zeros((self.T, N_tfeats), dtype='float32')


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # y and x are indexed directly
        y = self.d['y'][idx]
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        # indices of timestamps
        idx_clock = self.d['idx_clock'][idx] + self.counter

        # clock features
        x_clock = self.d['x_clock'][idx_clock]

        # fill in missing time feats with zeros
        if idx in self.d['tf']:
            x_tf = self.d['tf'][idx].reindex(
                index=range(self.T), fill_value=0).to_numpy()
        else:
            x_tf = self.tf0

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate((x_clock, x_tf, self.duration), axis=1)

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, x, x_time
        

    def __len__(self):
        return self.N_examples


    def collate(self, batch):
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