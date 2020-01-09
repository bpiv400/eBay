import numpy as np
from model.datasets.RecurrentDataset import RecurrentDataset


class DelayDataset(RecurrentDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a recurrent network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        :param sizes: dictionary of size parameters.
        '''
        super(DelayDataset, self).__init__(part, name, sizes)


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # periods is indexed directly
        periods = self.d['periods'][idx]

        # last index is 1 if offer arrives
        y = self.y0.copy()[:periods]
        if self.d['y'][idx]:
        	y[-1] = 1.0

        # initialize x from listing-level features
        x = self._construct_x(idx)

        # x_time
        x_time = self._construct_x_time(idx, periods)

        # add remaining to x_time
        remaining = self.d['remaining'][idx] - self.duration[:periods]
        x_time = np.concatenate((x_time, remaining), axis=1)

        return y, periods, x, x_time