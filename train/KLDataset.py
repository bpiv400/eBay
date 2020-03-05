import torch
import numpy as np
from train.EBayDataset import EBayDataset


class KLDataset(EBayDataset):
    def __init__(self, part, name):
        """
        Defines a parent class that extends torch.utils.data.Dataset.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        """
        super().__init__(part, name)

    def __getitem__(self, idx):
        """
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        """
        # y is indexed directly
        y = self.d['y'][idx]

        # index components of input dictionary
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        # p is indexed directly
        if 'turn' in self.d:
            turn = self.d['turn'][idx]
            p = self.d['p'][turn]
        else:
            p = self.d['p']

        return y, x, p

    @staticmethod
    def collate(batch):
        """
        Converts examples to tensors for a feed-forward network.
        :param batch: list of (dictionary of) numpy arrays.
        :return: dictionary of (dictionary of) tensors.
        """
        y, x, p = [], {}, []
        for b in batch:
            y.append(b[0])
            for k, v in b[1].items():
                if k in x:
                    x[k].append(torch.from_numpy(v))
                else:
                    x[k] = [torch.from_numpy(v)]
            p.append(torch.from_numpy(b[2]))

        # convert to (single) tensors
        y = torch.from_numpy(np.asarray(y)).long()
        x = {k: torch.stack(v).float() for k, v in x.items()}
        p = torch.stack(p).float()

        # output is dictionary of tensors
        return {'y': y, 'x': x, 'p': p}
