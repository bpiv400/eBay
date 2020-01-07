from torch.utils.data import Sampler, DataLoader
import numpy as np
import torch
from torch.nn.utils import rnn
from model.model_consts import MBSIZE, NUM_WORKERS


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


def collateFF(batch):
    '''
    Converts examples to tensors for a feed-forward network.
    :param batch: list of (dictionary of) numpy arrays.
    :return: dictionary of (dictionary of) tensors.
    '''
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


def collateRNN(batch):
    '''
    Converts examples to tensors for a recurrent network.
    :param batch: list of (dictionary of) numpy arrays.
    :return: dictionary of (dictionary of) tensors.
    '''
    y, periods, x, x_time = [], [], {}, []

    # sorts the batch list in decreasing order of periods
    for b in batch:
        y.append(torch.from_numpy(b[0]))
        periods.append(torch.as_tensor(b[1]))
        for k, v in b[2].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]
        x_time.append(torch.from_numpy(b[3]))

    # convert to tensor
    y = torch.stack(y).float()
    periods = torch.stack(periods).long()
    x = {k: torch.stack(v).float() for k, v in x.items()}
    x_time = torch.stack(x_time, dim=0).float()

    # pack for recurrent network
    x_time = rnn.pack_padded_sequence(
        x_time, periods, batch_first=True)

    # output is dictionary of tensors
    return {'y': y, 'x': x, 'x_time': x_time}


def get_batches(data, isTraining):
    '''
    Creates a Dataloader object.
    :param data: Inputs object.
    :param isTraining: chop into minibatches if True.
    :return: iterable batches of examples.
    '''
    batches = DataLoader(data, 
        collate_fn=collateFF if len(data.groups) == 1 else collateRNN,
        batch_sampler=Sample(data, isTraining),
        num_workers=NUM_WORKERS, 
        pin_memory=True)
    return batches