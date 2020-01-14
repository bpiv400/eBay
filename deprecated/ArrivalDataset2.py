import numpy as np
from compress_pickle import load
from model.datasets.eBayDataset import eBayDataset
from constants import INPUT_DIR, DAY
from featnames import TIME_FEATS


class ArrivalDataset(eBayDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a recurrent network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        :param sizes: dictionary of size parameters.
        '''
        super(ArrivalDataset, self).__init__(part, name, sizes)

        # save date feats lookup array to self
        self.date_feats = load(INPUT_DIR + 'date_feats.pkl')

        # number of labels
        self.N_labels = self.d['periods'][-1]

        # groups for sampling
        print('Creating groups...')
        self.groups = [np.array(range(self.N_labels), dtype='int64')]
        print('Done creating groups.')

        # save interval to self
        self.interval = sizes['interval']

        # array of zeros for time feats
        self.tf0 = np.zeros((len(TIME_FEATS),), dtype='float32')

        # maximum number of time steps
        T = sizes['interval_count']

        # period / max periods
        self.duration = np.expand_dims(
            np.array(range(T), dtype='float32') / T, axis=1)


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # initialize subprocess with hdf5 files
        if self.x is None:
            self._init_subprocess()

        # get the index of the lstg from periods
        idx_lstg = np.searchsorted(self.d['periods'], idx, side='right')

        # initialize x from listing-level features
        idx_x = self.d['idx_x'][idx_lstg]
        x = {k: v[idx_x, :] for k, v in self.x.items()}

        # number of periods since start of listing
        last = 0 if idx_lstg == 0 else self.d['periods'][idx_lstg-1]
        period = idx - last

        # clock features
        seconds = period * self.interval
        date_feats = self.date_feats[seconds // DAY]
        second_of_day = (seconds % DAY) / DAY
        x['clock'] = np.concatenate(
            (date_feats, second_of_day, self.duration[period]))

        # time features
        x['tf'] = self.tf0.copy()
        idx_tf = self.d['idx_tf'][idx_lstg]
        if idx_tf > -1:    # no time feats
            string = self.d['tf_periods'][idx_lstg].decode("utf-8")
            l = list(map(int, string.split('/')))
            if period >= l[0]:    # period is before 
                idx_last = np.searchsorted(l, period, side='left')
                x['tf'] = self.d['tf'][idx_tf + idx_last]

        # number of arrivals
        y = 0
        idx_arrival = self.d['idx_arrival'][idx_lstg]
        if idx_arrival > -1:
            string = self.d['arrival_periods'][idx_lstg].decode("utf-8")
            l = list(map(int, string.split('/')))
            if period in l:
                y = self.d['arrivals'][idx_arrival]
        
        return y, x


    def __len__(self):
        return self.N_labels