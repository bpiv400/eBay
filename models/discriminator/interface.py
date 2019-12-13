import sys
import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from datetime import datetime as dt
from torch.utils.data import DataLoader, Dataset, Sampler
from compress_pickle import load
from constants import *


# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):
    def __init__(self, part):

        # save data and parameters to self
        self.d = load('%s/inputs/%s/discriminator%s.gz' % (PREFIX, part))
        self.model = model

        # number of examples
        self.N = np.shape(self.d['lstg'])[0]


    def __getitem__(self, idx):
        return {k: v[idx, :] for k, v in self.d.items()}


    def __len__(self):
        return self.N


# defines a sampler that extends torch.utils.data.Sampler
class Sample(Sampler):
    def __init__(self, data, isTraining):
        """
        data: instance of Inputs
        isTraining: chop into minibatches if True
        """
        super().__init__(None)

        # chop into minibatches; shuffle for training
        v = [i for i in range(len(data))]
        if isTraining:
            np.random.shuffle(v)
        self.batches += np.array_split(v, 
            1 + len(v) // MBSIZE[isTraining])


    def __iter__(self):
        """
        Iterate over batches defined in intialization
        """
        for batch in self.batches:
            yield batch


    def __len__(self):
        return len(self.batches)


# helper function to run a loop of the model
def run_loop(simulator, data, optimizer=None):
    # training or validation
    isTraining = optimizer is not None

    # sampler
    sampler = Sample(data, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    loss = 0.0
    for b in batches:
        # multiplicative factor for regularization term in minibatch
        factor = float(b['lstg'].size()[0]) / data.N
        
        # move to device
        b = {k: v.to(simulator.device) for k, v in b.items()}

        # add minibatch loss
        loss += simulator.run_batch(b, factor, optimizer)

    return loss


# collate function for feedforward networks
def collateFF(batch):
    x = {}
    for b in batch:
        for k, v in b[1].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]

    # convert to (single) tensors and return
    return {k: torch.stack(v).float() for k, v in x.items()}