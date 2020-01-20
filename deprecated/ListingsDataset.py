import numpy as np
from model.datasets.FeedForwardDataset import FeedForwardDataset


class ListingsDataset(FeedForwardDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a feed-forward network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        super(ListingsDataset, self).__init__(part, name, sizes)

        # array of zeros for arrivals embedding
        self.arrival0 = np.zeros((sizes['x']['arrival'], ), dtype='float32')

    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # initialize subprocess with hdf5 files
        if self.x is None:
            self._init_subprocess()

        # y is indexed directly
        y = self.d['y'][idx]

        # initialize x from listing-level features
        x = self._construct_x(idx)

        # populate arrivals
        x['arrival'] = self._fill_array(self.arrival0, 'arrival', idx)

        return y, x