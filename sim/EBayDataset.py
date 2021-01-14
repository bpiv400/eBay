import numpy as np
from torch.utils.data import Dataset
from utils import load_file, load_inputs
from featnames import X_LSTG, LSTG, THREAD, DISCRIM_MODEL, VALIDATION, \
    TRAIN_MODELS, TRAIN_RL


class EBayDataset(Dataset):
    def __init__(self, name=None, train=None, part=None):
        """
        Defines a parent class that extends torch.utils.data.Dataset.
        :param str name: name of inputs file
        :param bool train: use valid partition if False
        :param str part: partition name
        """
        # save name to self
        self.name = name

        # partition
        if part is not None:
            self.part = part
        elif not train:
            self.part = VALIDATION
        else:
            if name == DISCRIM_MODEL:
                self.part = TRAIN_RL
            else:
                self.part = TRAIN_MODELS

        # listing features
        self.x_lstg = load_file(self.part, X_LSTG)

        # dictionary of inputs
        self.d = load_inputs(self.part, name)
        if 'x' in self.d:
            self.offer_keys = [k for k in self.d['x'].keys() if 'offer' in k]

        # number of labels
        self.N = len(self.d['y'])

    def __getitem__(self, idx):
        """
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        """
        # y is indexed directly
        y = self.d['y'][idx]

        # dictionary of listing components
        idx_x = self.d['idx_x'][idx]
        x = {k: v[idx_x, :] for k, v in self.x_lstg.items()}

        # add in thread and offer features
        if 'x' in self.d:
            x_thread = self.d['x'][THREAD][idx, :]
            x[LSTG] = np.concatenate([x[LSTG], x_thread])
            if len(self.offer_keys) > 0:
                for k in self.offer_keys:
                    x[k] = self.d['x'][k][idx, :]

        return y, x

    def __len__(self):
        return self.N
