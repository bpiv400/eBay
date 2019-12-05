import sys
import numpy as np, pandas as pd
import torch
from torch.nn.utils import rnn
from datetime import datetime as dt
from torch.utils.data import DataLoader, Dataset, Sampler
from compress_pickle import load
from constants import *


# helper function to run a loop of the model
def run_loop(simulator, data, optimizer=None):
    # training or validation
    isTraining = optimizer is not None

    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # sampler
    sampler = Sample(data, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0.0
    for batch in batches:
        lnL += simulator.run_batch(batch, optimizer)

    return lnL / data.N_labels



# defines a dataset that extends torch.utils.data.Dataset
class Inputs(Dataset):

    def __init__(self, part, model):

        # save data and parameters to self
        self.d = load('%s/inputs/%s/%s.gz' % (PREFIX, part, model))
        self.model = model
        self.isRecurrent = len(np.shape(self.d['y'])) > 1

        # number of examples
        self.N = np.shape(self.d['y'])[0]
        
        # number of labels, for normalizing loss
        if self.isRecurrent:
            self.N_labels = np.sum(self.d['y'] > -1)
        else:
            self.N_labels = self.N

        # for arrival and delay models
        if 'tf' in self.d:
            # number of time steps
            self.n = np.shape(self.d['y'])[1]

            # counter for expanding clock features
            role = model.split('_')[-1]
            interval = int(INTERVAL[role] / 60) # interval in minutes
            self.counter = interval * np.array(range(self.n), dtype='uint16')

            # period / max periods
            self.duration = np.expand_dims(
                np.array(range(self.n), dtype='float32') / self.n, axis=1)

            # number of time features
            for val in self.d['tf'].values():
                N_tfeats = len(val.columns)
                break
                
            # empty time feats
            self.tf0 = np.zeros((self.n, N_tfeats), dtype='float32')


    def __getitem__(self, idx):
        # all models index y using idx
        y = self.d['y'][idx]

        # index in dictionary
        x = {k: v[idx, :] for k, v in self.d['x'].items()}

        # feed-forward models
        if not self.isRecurrent:
            return y, x

        # indices of timestamps
        idx_clock = self.d['idx_clock'][idx] + self.counter

        # clock features
        x_clock = self.d['x_clock'][idx_clock].astype('float32')

        # fill in missing time feats with zeros
        if idx in self.d['tf']:
            x_tf = self.d['tf'][idx].reindex(
                index=range(self.n), fill_value=0).to_numpy()
        else:
            x_tf = self.tf0

        # time feats: first clock feats, then time-varying feats
        x_time = np.concatenate((x_clock, x_tf, self.duration), axis=1)

        # for delay models, add (normalized) periods remaining
        if 'remaining' in self.d:
            remaining = self.d['remaining'][idx] - self.duration
            x_time = np.concatenate((x_time, remaining), axis=1)

        return y, x, x_time


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
        self.batches = []
        for v in data.d['groups']:
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


# collate function for feedforward networks
def collateFF(batch):
    y, x = [], {}
    for b in batch:
        y.append(b[0])
        for k, v in b[1].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]

    # convert to tensor
    y = torch.from_numpy(np.asarray(y))
    x = {k: torch.stack(v).float() for k, v in x.items()}

    # output is dictionary of tensors
    return {'y': y, 'x': x}


# collate function for recurrent networks
def collateRNN(batch):
    # initialize output
    y, x, x_time = [], {}, []

    # sorts the batch list in decreasing order of turns
    for b in batch:
        y.append(torch.from_numpy(b[0]))
        for k, v in b[1].items():
            if k in x:
                x[k].append(torch.from_numpy(v))
            else:
                x[k] = [torch.from_numpy(v)]
        x_time.append(torch.from_numpy(b[2]))

    # convert to tensor, pack if needed
    y = torch.stack(y).float()
    turns = torch.sum(y > -1, dim=1).long()
    x = {k: torch.stack(v).float() for k, v in x.items()}
    x_time = torch.stack(x_time, dim=0).float()
    x_time = rnn.pack_padded_sequence(x_time, turns, batch_first=True)
    
    # output is dictionary of tensors
    return {'y': y, 
            'x': x, 
            'x_time': x_time}
