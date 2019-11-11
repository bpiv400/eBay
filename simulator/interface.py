import sys
import numpy as np, pandas as pd
import torch
from datetime import datetime as dt
from torch.utils.data import Dataset, Sampler
from compress_pickle import load
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, part, model):

        # inputs
        self.d = load('%s/inputs/%s/%s.gz' % (PREFIX, part, model))

        # number of examples
        self.N = np.shape(self.d['x_fixed'])[0]
        self.model = model
        self.isRecurrent = 'turns' in self.d

        # number of labels, for normalizing loss
        if self.isRecurrent:
            self.N_labels = np.sum(self.d['turns'])
        else:
            self.N_labels = self.N

        # for arrival and delay models
        if 'tf' in self.d:
            # number of time steps
            self.n = self.d['turns'][0]
            # interval for clock features
            role = model.split('_')[-1]
            interval = int(INTERVAL[role] / 60) # interval in minutes
            self.counter = interval * np.array(range(self.n), dtype='uint16')
            # period / max periods
            self.duration = np.expand_dims(
                np.array(range(self.n) / self.n, dtype='float32'), axis=1)
            # number of time features
            for val in self.d['tf'].values():
                N_tfeats = len(val.columns)
                break
            # empty time feats
            self.tf0 = np.zeros((self.n, N_tfeats), dtype='float32')


    def __getitem__(self, idx):
        # all models index y and x_fixed using idx
        y = self.d['y'][idx]
        x_fixed = self.d['x_fixed'][idx,:]

        # feed-forward models
        if not self.isRecurrent:
            return y, x_fixed, idx

        # number of turns
        turns = self.d['turns'][idx]

        # x_time
        if 'x_time' in self.d:
            x_time = self.d['x_time'][idx,:,:]
        else:
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
        
        return y, turns, x_fixed, x_time, idx


    def __len__(self):
        return self.N


# defines a sampler that extends torch.utils.data.Sampler
class Sample(Sampler):

    def __init__(self, data, isTraining):
        """
        data: instance of Inputs
        isTraining: chop into minibatches if True
        """
        super().__init__(None)

        # chop into minibatches; shuffle for training
        self.batches = []
        for v in data.d['groups']:
            if isTraining:
                np.random.shuffle(v)
            self.batches += np.array_split(v, 1 + len(v) // MBSIZE)
        # shuffle training batches
        if isTraining:
            np.random.shuffle(self.batches)


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
    y, x_fixed, idx = [], [], []
    for b in batch:
        y.append(b[0])
        x_fixed.append(torch.from_numpy(b[1]))
        idx.append(b[2])

    # convert to tensor
    y = torch.from_numpy(np.asarray(y))
    x_fixed = torch.stack(x_fixed).float()
    idx = torch.tensor(idx)

    # output is dictionary of tensors
    return {'y': y, 'x_fixed': x_fixed}


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, turns, x_fixed, x_time, idx = [], [], [], [], []

    # sorts the batch list in decreasing order of turns
    for b in batch:
        y.append(torch.from_numpy(b[0]))
        turns.append(b[1])
        x_fixed.append(torch.from_numpy(b[2]))
        x_time.append(torch.from_numpy(b[3]))
        idx.append(b[4])

    # convert to tensor, pack if needed
    y = torch.stack(y)
    turns = torch.from_numpy(np.asarray(
        turns, dtype='int64')).long()
    x_fixed = torch.stack(x_fixed).float()
    x_time = torch.stack(x_time, dim=0).float()
    idx = torch.tensor(idx)
    
    # output is dictionary of tensors
    return {'y': y, 
            'turns': turns,
            'x_fixed': x_fixed, 
            'x_time': x_time}
