import numpy as np
from model.datasets.eBayDataset import eBayDataset


class FeedForwardDataset(eBayDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a feed-forward network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        super(FeedForwardDataset, self).__init__(part, name, sizes)

        # load outcome into memory
        self.y = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # number of labels
        self.N_labels = len(self.y)

        # groups for sampling
        self.groups = [np.array(range(self.N_labels))]


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
        y = self.y[idx]

        # initialize x from listing-level features
        x = self._construct_x(idx)

        return y, x


    def __len__(self):
        return self.N_labels