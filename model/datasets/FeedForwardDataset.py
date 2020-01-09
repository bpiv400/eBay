import numpy as np
from torch.utils.data import Dataset
from model.datasets.eBayDataset import eBayDataset


class FeedForwardDataset(eBayDataset):
    def __init__(self, part, name):
        '''
        Defines a child class of eBayDataset for a feed-forward network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        super(FeedForwardDataset, self).__init__(part, name)

        # number of labels
        self.N_labels = np.shape(self.d['y'])[0]

        # create single group for sampling
        self.groups = [np.array(range(self.N_labels))]


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # y is indexed directly
        y = self.d['y'][idx]

        # initialize x from listing-level features
        x = self._construct_x(idx)

        return y, x