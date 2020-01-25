import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader, Dataset
from train.train_consts import MBSIZE, NUM_WORKERS


class PartialDataset(Dataset):
    def __init__(self, x):
        '''
        Defines a parent class that extends torch.utils.data.Dataset.
        :param part: string partition name (e.g., train_models).
        :param name: string model name.
        '''
        # save name to self
        self.x = x

        # number of labels
        self.N = len(self.x['lstg'])


    def __getitem__(self, idx):
        '''
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        '''
        # index components of input dictionary
        return {k: v[idx, :] for k, v in self.x.items()}


    def __len__(self):
        return self.N



class Sample(Sampler):
    def __init__(self, data):
        '''
        Defines a sampler that extends torch.utils.data.Sampler.
        :param data: Inputs object.
        '''
        super().__init__(None)

        # create vector of indices
        v = np.array(range(len(data)))

        # chop into minibatches
        self.batches = np.array_split(v, 
            1 + len(v) // (MBSIZE[False] * 15))


    def __iter__(self):
        """
        Iterate over batches defined in intialization
        """
        for batch in self.batches:
            yield batch


    def __len__(self):
        return len(self.batches)


def collate(batch):
    '''
    Converts examples to tensors for a feed-forward network.
    :param batch: list of (dictionary of) numpy arrays.
    :return: dictionary of (dictionary of) tensors.
    '''
    x = {}
    for b in batch:
        for k, v in b.items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]

    # convert to (single) tensors
    return {k: torch.stack(v).float() for k, v in x.items()}


def get_batches(data):
    '''
    Creates a Dataloader object.
    :param data: Inputs object.
    :param isTraining: chop into minibatches if True.
    :return: iterable batches of examples.
    '''
    batches = DataLoader(data, collate_fn=collate,
        batch_sampler=Sample(data),
        num_workers=8, pin_memory=True)
    return batches