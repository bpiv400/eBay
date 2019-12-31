import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset
from compress_pickle import load
from constants import PARTS_DIR, INPUT_DIR, INTERVAL, INTERVAL_COUNTS, DAY


class RecurrentDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a dataset that extends torch.utils.data.Dataset
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # save clock feats lookup array to self
        self.date_feats = load(INPUT_DIR + 'date_feats.pkl')

        # groups for sampling
        self.groups = [np.nonzero(self.d['periods'] == n)[0] \
            for n in np.unique(self.d['periods'])]

        # number of examples and labels
        self.N_examples = np.shape(self.d['periods'])[0]
        self.N_labels = np.sum(self.d['periods'])

        # number of time steps
        role = name.split('_')[-1]
        T = INTERVAL_COUNTS[role]

        # counter for expanding clock features
        self.counter = INTERVAL[role] * np.array(range(T))

        # period / max periods
        self.duration = np.expand_dims(
            np.array(range(T), dtype='float32') / T, axis=1)

        # empty time feats
        for val in self.d['tf'].values():
            N_tfeats = len(val.columns)
            break
        self.tf0 = np.zeros((T, N_tfeats), dtype='float32')

        # outcome of zeros
        self.y0 = np.zeros((T, ), dtype='int8')


    def _create_y(self, idx, periods=None):
        raise NotImplementedError()


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # periods is indexed directly
        periods = self.d['periods'][idx]

        # components of x are indexed directly
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        # y gets reindexed
        y = self._create_y(idx, periods)

        # indices of timestamps
        seconds = self.d['seconds'][idx] + self.counter

        # clock features
        date_feats = self.date_feats[seconds // DAY]
        second_of_day = np.expand_dims((seconds % DAY) / DAY, axis=1)
        clock_feats = np.concatenate(
            (date_feats, second_of_day), axis=1)

        # fill in missing time feats with zeros
        if idx in self.d['tf']:
            x_tf = self.d['tf'][idx].reindex(
                index=range(periods), fill_value=0).to_numpy()
        else:
            x_tf = self.tf0.copy()[:periods, :]

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate(
            (clock_feats, x_tf, self.duration), axis=1)

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, periods, x, x_time
        

    def __len__(self):
        return self.N_examples


    def collate(self, batch):
        '''
        Converts examples to tensors for a recurrent network.
        :param batch: list of (dictionary of) numpy arrays.
        :return: dictionary of (dictionary of) tensors.
        '''
        y, periods, x, x_time = [], [], {}, []

        # sorts the batch list in decreasing order of periods
        for b in batch:
            y.append(torch.from_numpy(b[0]))
            periods.append(torch.from_numpy(b[1]))
            for k, v in b[2].items():
                if k in x:
                    x[k].append(torch.from_numpy(v))
                else:
                    x[k] = [torch.from_numpy(v)]
            x_time.append(torch.from_numpy(b[3]))

        # convert to tensor
        y = torch.stack(y).float()
        periods = torch.stack(periods).long()
        x = {k: torch.stack(v).float() for k, v in x.items()}
        x_time = torch.stack(x_time, dim=0).float()

        # pack for recurrent network
        x_time = rnn.pack_padded_sequence(
            x_time, periods, batch_first=True)

        # output is dictionary of tensors
        return {'y': y, 'x': x, 'x_time': x_time}


class ArrivalDataset(RecurrentDataset):
    def __init__(self, part, name):
        super(ArrivalDataset, self).__init__(part, name)

        # load lstg features and convert to to numpy
        self.d['x'] = load(PARTS_DIR + '{}/x_lstg.gz'.format(part))
        self.d['x'] = {k: v.numpy() for k, v in self.d['x'].items()}

    def _create_y(self, idx, periods):
        # if at least one arrival is observed
        if idx in self.d['y']:
            return self.d['y'][idx].reindex(
                index=range(periods), fill_value=0).to_numpy()
        
        # no arrivals
        return self.y0.copy()[:periods]


class DelayDataset(RecurrentDataset):
    def __init__(self, part, name):
        super(ArrivalDataset, self).__init__(part, name)

    def _create_y(self, idx, periods):
        # initialize periods-length aarry of zeros
        y = self.y0.copy()[:periods]

        # offer arrives in final period
        if self.d['y'][idx]:
            y[periods-1] = 1

        return y