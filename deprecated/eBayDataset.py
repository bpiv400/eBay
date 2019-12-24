import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
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