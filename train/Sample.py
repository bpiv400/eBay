import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from train.train_consts import MBSIZE, NUM_WORKERS


class Sample(Sampler):
    def __init__(self, data, is_training):
        """
        Defines a sampler that extends torch.utils.data.Sampler.
        :param data: Inputs object.
        :param is_training: chop into minibatches if True.
        """
        super().__init__(None)

        # create vector of indices
        v = np.array(range(len(data)))
        if is_training:
            np.random.shuffle(v)

        # chop into minibatches
        self.batches = np.array_split(v, 1 + len(v) // MBSIZE[is_training])

    def __iter__(self):
        """
        Iterate over batches defined in intialization
        """
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def get_batches(data, is_training=False):
    """
    Creates a Dataloader object.
    :param data: Inputs object.
    :param is_training: chop into minibatches if True.
    :return: iterable batches of examples.
    """
    batches = DataLoader(data, collate_fn=data.collate,
                         batch_sampler=Sample(data, is_training),
                         num_workers=NUM_WORKERS[data.name],
                         pin_memory=True)
    return batches
