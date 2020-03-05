from compress_pickle import load
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, DataLoader, Dataset
from train.train_consts import MBSIZE
from train.EBayDataset import EBayDataset
from nets.FeedForward import FeedForward
from constants import INPUT_DIR, MODEL_DIR, INDEX_DIR


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


def get_role_outcomes(part, name):
    # create model
    sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
    net = FeedForward(sizes)
    model_path = MODEL_DIR + '{}.net'.format(name)
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)

    # make predictions for each example
    data = EBayDataset(part, name)
    with torch.no_grad():
        theta = net(data)

    # pandas index
    idx = load(INDEX_DIR + '{}/{}.gz'.format(part, name))

    # convert to distribution
    if outcome == 'msg':
        p_hat = torch.sigmoid(theta)
        p_hat = pd.Series(p_hat.numpy(), index=idx)
    else:
        p_hat = torch.exp(torch.nn.functional.log_softmax(theta, dim=-1))
        p_hat = pd.DataFrame(p_hat.numpy(),
                             index=idx,
                             columns=range(p_hat.size()[1]))

    # observed outcomes
    y = pd.Series(data.d['y'], index=idx)

    return y, p_hat


def get_outcomes(outcome):
    if outcome in ['delay', 'con', 'msg']:
        # outcomes by role
        y_byr, p_hat_byr = get_outcomes('%s_byr' % outcome)
        y_slr, p_hat_slr = get_outcomes('%s_slr' % outcome)

        # combine
        y = pd.concat([y_byr, y_slr], dim=0)
        p_hat = pd.concat([p_hat_byr, p_hat_slr], dim=0)

    # no roles for arrival and hist models
    else:
        y, p_hat = get_role_outcomes(outcome)

    # sort and return
    return y.sort_index(), p_hat.sort_index()
