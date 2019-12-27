from torch.utils.data import Sampler, DataLoader
import numpy as np
from model.consts import *


class Sample(Sampler):
    def __init__(self, data, isTraining):
        '''
        Defines a sampler that extends torch.utils.data.Sampler.
        :param data: Inputs object.
        :param isTraining: chop into minibatches if True.
        '''
        super().__init__(None)

        # chop into minibatches; shuffle for training
        self.batches = []
        for v in data.groups:
            if isTraining:
                np.random.shuffle(v)
            self.batches += np.array_split(v, 
                1 + len(v) // MBSIZE[isTraining])
        # shuffle training batches
        if isTraining:
            np.random.shuffle(self.batches)


    def __iter__(self):
        """
        Iterate over batches defined in intialization
        """
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def get_batches(data, isTraining):
    '''
    Creates a Dataloader object.
    :param data: Inputs object.
    :param isTraining: chop into minibatches if True.
    :return: iterable batches of examples.
    '''
    batches = DataLoader(data, collate_fn=data.collate,
        batch_sampler=Sample(data, isTraining),
        num_workers=NUM_WORKERS, pin_memory=True)
    return batches