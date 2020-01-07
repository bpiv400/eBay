import numpy as np
from torch.utils.data import Dataset
from compress_pickle import load
from constants import INPUT_DIR, DAY
from featnames import TIME_FEATS


class RecurrentDataset(Dataset):
    def __init__(self, part, name, sizes):
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
        self.N_examples = len(self.d['periods'])
        self.N_labels = np.sum(self.d['periods'])

        # maximum number of time steps
        T = sizes['interval_count']

        # counter for expanding clock features
        self.counter = sizes['interval'] * np.array(range(T))

        # period / max periods
        self.duration = np.expand_dims(
            np.array(range(T), dtype='float32') / T, axis=1)

        # default outcome and time feats
        self.y0 = np.zeros((T, ), dtype='float32')
        self.tf0 = np.zeros((T, len(TIME_FEATS)), dtype='float32')


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # components that are indexed directly
        periods = self.d['periods'][idx]
        x = {k: v[idx] for k, v in self.d['x'].items()}

        # y gets reindexed
        y = self.y0.copy()[:periods]
        for k, v in self.d['y'][idx].items():
            y[k] = v
            
        # indices of timestamps
        seconds = self.d['seconds'][idx] + self.counter[:periods]

        # clock features
        date_feats = self.date_feats[seconds // DAY]
        second_of_day = np.expand_dims((seconds % DAY) / DAY, axis=1)
        clock_feats = np.concatenate(
            (date_feats, second_of_day), axis=1)

        # fill in missing time feats with zeros
        x_tf = self.tf0.copy()[:periods, :]
        for k, v in self.d['tf'][idx].items():
            if k < periods:
                x_tf[k, :] = v

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate(
            (clock_feats, x_tf, self.duration[:periods]), axis=1)

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration[:periods]
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, periods, x, x_time
        

    def __len__(self):
        return self.N_examples
