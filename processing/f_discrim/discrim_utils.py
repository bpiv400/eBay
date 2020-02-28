import torch
import numpy as np
import pandas as pd
from compress_pickle import load, dump
from torch.utils.data import Sampler, DataLoader, Dataset
from train.train_consts import MBSIZE
from processing.processing_utils import save_featnames, save_sizes, convert_x_to_numpy
from constants import SIM_CHUNKS, ENV_SIM_DIR, MAX_DELAY, ARRIVAL_PREFIX, INDEX_DIR


class PartialDataset(Dataset):
    def __init__(self, x):
        """
        Defines a parent class that extends torch.utils.data.Dataset.
        :param x: dictionary of numpy arrays of input data.
        """
        # save name to self
        self.x = x

        # number of labels
        self.N = len(self.x['lstg'])

    def __getitem__(self, idx):
        """
        Returns a tuple of data components for example.
        :param idx: index of example.
        :return: tuple of data components at index idx.
        """
        # index components of input dictionary
        return {k: v[idx, :] for k, v in self.x.items()}

    def __len__(self):
        return self.N


class Sample(Sampler):
    def __init__(self, data):
        """
        Defines a sampler that extends torch.utils.data.Sampler.
        :param data: Inputs object.
        """
        super().__init__(None)

        # create vector of indices
        v = np.array(range(len(data)))

        # chop into minibatches
        self.batches = np.array_split(v, 1 + len(v) // (MBSIZE[False] * 15))

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
    """
    Creates a Dataloader object.
    :param data: Inputs object.
    :return: iterable batches of examples.
    """
    batches = DataLoader(data, collate_fn=collate,
                         batch_sampler=Sample(data),
                         num_workers=8,
                         pin_memory=True)
    return batches


def process_lstg_end(lstg_start, lstg_end):
    # remove thread and index from lstg_end index
    lstg_end = lstg_end.reset_index(['thread', 'index'], drop=True)
    assert not lstg_end.index.duplicated().max()

    # fill in missing lstg end times with expirations
    lstg_end = lstg_end.reindex(index=lstg_start.index, fill_value=-1)
    lstg_end.loc[lstg_end == -1] = lstg_start + MAX_DELAY[ARRIVAL_PREFIX] - 1

    return lstg_end


def get_sim_times(part, lstg_start):
    # collect times from simulation files
    lstg_end, thread_start = [], []
    for i in range(1, SIM_CHUNKS+1):
        sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
        offers, threads = [sim[k] for k in ['offers', 'threads']]
        lstg_end.append(offers.loc[(offers.con == 100) & ~offers.censored, 'clock'])
        thread_start.append(threads.clock)

    # concatenate into single series
    lstg_end = pd.concat(lstg_end, axis=0).sort_index()
    thread_start = pd.concat(thread_start, axis=0).sort_index()

    # shorten index and fill-in expirations
    lstg_end = process_lstg_end(lstg_start, lstg_end)

    return lstg_end, thread_start


def save_discrim_files(part, name, x_obs, x_sim):
    # featnames and sizes
    if part == 'test_rl':
        save_featnames(x_obs, name)
        save_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['lstg'].index
    idx_sim = x_sim['lstg'].index

    # save joined index
    idx_joined = idx_obs.union(idx_sim, sort=False)
    dump(idx_joined, INDEX_DIR + '{}/listings.gz'.format(part))

    # create dictionary of numpy arrays
    x_obs = convert_x_to_numpy(x_obs, idx_obs)
    x_sim = convert_x_to_numpy(x_sim, idx_sim)

    # y=1 for real data
    y_obs = np.ones(len(idx_obs), dtype=bool)
    y_sim = np.zeros(len(idx_sim), dtype=bool)
    d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

    # join input variables
    assert x_obs.keys() == x_sim.keys()
    d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0) for k in x_obs.keys()}

    # save inputs
    dump(d, INPUT_DIR + '{}/listings.gz'.format(part))

    # save small
    if part == 'train_rl':
        save_small_discrim()
