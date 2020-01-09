from model.datasets.RecurrentDataset import RecurrentDataset


class ArrivalDataset(RecurrentDataset):
    def __init__(self, part, name, sizes):
        '''
        Defines a child class of eBayDataset for a recurrent network.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        :param sizes: dictionary of size parameters.
        '''
        super(ArrivalDataset, self).__init__(part, name, sizes)


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # periods is indexed directly
        periods = self.d['periods'][idx]

        # initialize x from listing-level features
        x = self._construct_x(idx)

        # x_time
        x_time = self._construct_x_time(idx, periods)

        # non-zero arrivals get added to vector of zeros
        y = self.y0.copy()[:periods]
        for k, v in self.d['y'][idx].items():
            y[k] = v

        return y, periods, x, x_time