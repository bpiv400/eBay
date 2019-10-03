import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
import h5py, pickle


# Converts data frames to tensors sorted (descending) by N_turns.
def convert_to_tensors(d):
    # for feed-forward networks
    if 'x_time' not in d:
        d['x_fixed'] = torch.tensor(d['x_fixed'].reindex(
            d['y'].index).astype(np.float32).values)
        d['y'] = torch.tensor(d['y'].astype(np.float32).values)

    # for recurrent networks
    else:
        # sorting index
        idxnames = d['y'].index.names
        s = d['y'].groupby(idxnames[:-1]).transform('count').rename('count')
        s = s.reset_index().sort_values(['count'] + idxnames,
            ascending=[False] + [True for i in range(len(idxnames))])
        s = s.set_index(idxnames).squeeze()

        # number of turns
        turns = d['y'].groupby(idxnames[:-1]).count()
        turns = turns.sort_values(ascending=False)
        d['turns'] = torch.tensor(turns.values)

        # outcome
        d['y'] = torch.tensor(np.transpose(d['y'].unstack().reindex(
            index=turns.index).astype(np.float32).values))

        # fixed features
        d['x_fixed'] = torch.tensor(d['x_fixed'].reindex(
            index=turns.index).astype(np.float32).values).unsqueeze(dim=0)

        # timestep features
        arrays = []
        for c in d['x_time'].columns:
            array = d['x_time'][c].astype(np.float32).unstack().reindex(
                index=turns.index).values
            arrays.append(np.expand_dims(np.transpose(array), axis=2))
        d['x_time'] = torch.tensor(np.concatenate(arrays, axis=2))

    return d


# Defines a data loader that extends torch.utils.data.Dataset
class DataLoader(Dataset):

    def __init__(self, partition, model, outcome):

        # save parameters to self
        self.partition = partition
        self.model = model
        self.outcome = outcome

        # path
        path = 'data/inputs/%s/%s_%s.hdf5' % (partition, model, outcome)

        # load data file
        self.d = h5py.File(path, 'r')

        # extract dictionary components
        self.y, self.x_fixed = [self.d[key] for key in ['y', 'x_fixed']]
        if self.model != 'arrival':
            self.x_time, self.turns = [self.d[key] for key in ['x_time', 'turns']]

        # save length
        self.N = np.shape(self.x_fixed)[0]


    def close(self):
        self.d.close()


    def __getitem__(self, idx):
        # create dictionary
        d = {'y': self.d['y'][idx],
             'x_fixed': self.d['x_fixed'][idx,:]}

        if self.model != 'arrival':
            d['x_time'] = self.d['x_time'][:,idx,:]
            d['turns'] = self.d['turns'][idx]

        # convert to tensors
        d = convert_to_tensors(d)

    def __len__(self):
        return self.N