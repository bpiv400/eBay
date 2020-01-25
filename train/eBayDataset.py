from torch.utils.data import Dataset
from compress_pickle import load
from constants import INPUT_DIR


class eBayDataset(Dataset):
    def __init__(self, part, name):
        '''
        Defines a parent class that extends torch.utils.data.Dataset.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        # save name to self
        self.name = name

        # dictionary of inputs        
        self.d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))

        # number of labels
        self.N = len(self.d['y'])


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # y is indexed directly
        y = self.d['y'][idx]

        # index components of input dictionary
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        return y, x
        

    def __len__(self):
        return self.N