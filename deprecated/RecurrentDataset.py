import numpy as np
from compress_pickle import load
from model.datasets.eBayDataset import eBayDataset
from constants import INPUT_DIR, DAY
from featnames import TIME_FEATS


class RecurrentDataset(eBayDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a recurrent network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        :param sizes: dictionary of size parameters.
        '''
        super(RecurrentDataset, self).__init__(part, name, sizes)

        # load number of periods
        self.periods = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # number of labels and examples
        self.N_labels = np.sum(self.periods)
        self.N_examples = len(self.periods)

        # groups for sampling
        self.groups = [np.nonzero(self.periods == n)[0] \
            for n in np.unique(self.periods)]

        # save date feats lookup array to self
        self.date_feats = load(INPUT_DIR + 'date_feats.pkl')

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


    def _construct_x_time(self, idx, periods):
        # indices of timestamps
        seconds = self.d['seconds'][idx] + self.counter[:periods]

        # clock features
        date_feats = self.date_feats[seconds // DAY]
        second_of_day = np.expand_dims((seconds % DAY) / DAY, axis=1)
        clock_feats = np.concatenate(
            (date_feats, second_of_day), axis=1)

        # fill in missing time feats with zeros
        x_tf = self.tf0.copy()[:periods, :]
        x_tf = self._fill_array(x_tf, 'tf', idx)

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate(
            (clock_feats, x_tf, self.duration[:periods]), axis=1)

        return x_time


    def __getitem__(self, idx):
        raise NotImplementedError()


    def __len__(self):
        return self.N_examples
