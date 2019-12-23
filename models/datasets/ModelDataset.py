import numpy as np, pandas as pd
from models.datasets.eBayDataset import eBayDataset
from constants import *


class ModelDataset(eBayDataset):
    def __init__(self, part, name):
        super().__init__(part, name)

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
