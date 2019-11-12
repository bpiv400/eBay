import sys
import numpy as np, pandas as pd
import torch
from datetime import datetime as dt
from torch.utils.data import Dataset
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, d, model):

        # save data and parameters to self
        self.d = d
        self.model = model
        self.isRecurrent = 'turns' in self.d

        # number of examples
        self.N = np.shape(self.d['x_fixed'])[0]
        
        # number of labels, for normalizing loss
        if self.isRecurrent:
            self.N_labels = np.sum(self.d['turns'])
        else:
            self.N_labels = self.N

        # for arrival and delay models
        if 'tf' in self.d:
            # number of time steps
            self.n = np.shape(self.d['y'])[1]
            # interval for clock features
            role = model.split('_')[-1]
            interval = int(INTERVAL[role] / 60) # interval in minutes
            self.counter = interval * np.array(
                range(self.n), dtype='uint16')
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

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, turns, x_fixed, x_time, idx


    def __len__(self):
        return self.N