import psutil
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from constants import FIRST_ARRIVAL_MODEL, BYR_DELAY_POLICY_INIT, \
    BYR_DELAY_VALUE_INIT

NUM_WORKERS = {FIRST_ARRIVAL_MODEL: 5,
               BYR_DELAY_POLICY_INIT: 2,
               BYR_DELAY_VALUE_INIT: 2}
MBSIZE = {True: 128, False: 1e4}  # True for training, False for validation


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


def collate(batch):
    """
    Converts examples to tensors for a feed-forward network.
    :param batch: list of (dictionary of) numpy arrays.
    :return: dictionary of (dictionary of) tensors.
    """
    y, x = [], {}
    for b in batch:
        y.append(b[0])
        for k, v in b[1].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]

    # convert to (single) tensors
    y = torch.from_numpy(np.asarray(y)).long()
    x = {k: torch.stack(v).float() for k, v in x.items()}

    # output is dictionary of tensors
    return {'y': y, 'x': x}


def get_batches(data, is_training=False):
    """
    Creates a Dataloader object.
    :param data: Inputs object.
    :param is_training: chop into minibatches if True.
    :return: iterable batches of examples.
    """
    if data.name in NUM_WORKERS:
        num_workers = NUM_WORKERS[data.name]
    else:
        num_workers = len(psutil.Process().cpu_affinity())
    batches = DataLoader(data,
                         collate_fn=collate,
                         batch_sampler=Sample(data, is_training),
                         num_workers=num_workers,
                         pin_memory=True)
    return batches
