import math
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from constants import FIRST_ARRIVAL_MODEL
from utils import load_model
from agent.agent_consts import NO_ARRIVAL, NO_ARRIVAL_CUTOFF
from rlenv.env_utils import proper_squeeze
from rlenv.Composer import Composer


def get_no_arrival_likelihood(x_lstg_chunk=None, model=None):
    for key, val in x_lstg_chunk.items():
        x_lstg_chunk[key] = torch.from_numpy(x_lstg_chunk[key]).float()
    logits = model(x_lstg_chunk)
    pi = softmax(logits, dim=logits.dim() - 1)
    pi = pi[:, pi.shape[1] - 1]
    return pi


def get_batch_unlikely(x_lstg_chunk=None, model=None):
    pi = get_no_arrival_likelihood(x_lstg_chunk=x_lstg_chunk, model=model)
    unlikely = torch.nonzero(pi > NO_ARRIVAL_CUTOFF)
    return unlikely


def process_unlikely(unlikely=None, start_idx=None):
    unlikely + start_idx
    unlikely = proper_squeeze(unlikely)
    unlikely = unlikely.tolist()
    return unlikely


def chunk_x_lstg(i=None, batch_size=None, composer=None, x_lstg=None):
    start_idx = i * batch_size
    end_idx = min(start_idx + batch_size, len(x_lstg.index))
    x_lstg_chunk = x_lstg.iloc[start_idx:end_idx]
    x_lstg_chunk = composer.decompose_x_lstg(x_lstg=x_lstg_chunk)
    return x_lstg_chunk


def lookup_x_lstg_comparison(x_lstg=None, lookup=None):
    assert x_lstg.index.name == lookup.index.name
    assert lookup.index.name == 'lstg'
    assert lookup.index.equals(x_lstg.index)


def add_no_arrival_likelihood(x_lstg=None, lookup=None):
    """
    Removes listings from x_lstg and lookup DataFrames
    that fall under some minimum threshold for likelihood
    of arrival in the first year

    :param pd.DataFrame x_lstg:
    :param pd.DataFrame lookup:
    :return: (x_lstg, lookup)
    """
    # error checking
    lookup_x_lstg_comparison(x_lstg=x_lstg, lookup=lookup)

    # setup search
    batch_size = 1024.0
    model = load_model(FIRST_ARRIVAL_MODEL)
    composer = Composer(x_lstg.columns)
    chunks = math.ceil(len(x_lstg.index) / batch_size)
    # iterate over chunks
    all_pi = list()
    for i in range(chunks):
        x_lstg_chunk = chunk_x_lstg(i=i, batch_size=int(batch_size),
                                    composer=composer, x_lstg=x_lstg)
        pi = get_no_arrival_likelihood(x_lstg_chunk=x_lstg_chunk,
                                       model=model).numpy()
        all_pi.append(pi)
    all_pi = np.concatenate(all_pi)
    assert len(all_pi.shape) == 1
    lookup[NO_ARRIVAL] = all_pi
    return lookup


def remove_unlikely_arrival_lstgs(x_lstg=None, lookup=None):
    """
    Removes listings from x_lstg and lookup DataFrames
    that fall under some minimum threshold for likelihood
    of arrival in the first year

    :param pd.DataFrame x_lstg:
    :param pd.DataFrame lookup:
    :return: (x_lstg, lookup)
    """
    # error checking
    lookup_x_lstg_comparison(x_lstg=x_lstg, lookup=lookup)
    org_length = len(lookup.index)

    # setup search
    batch_size = 1024.0
    model = load_model(FIRST_ARRIVAL_MODEL)
    composer = Composer(x_lstg.columns)
    chunks = math.ceil(len(x_lstg.index) / batch_size)
    unlikely_lstgs = list()
    # iterate over chunks
    for i in range(chunks):
        x_lstg_chunk = chunk_x_lstg(i=i, batch_size=int(batch_size),
                                    composer=composer, x_lstg=x_lstg)
        unlikely = get_batch_unlikely(x_lstg_chunk=x_lstg_chunk, model=model)
        if unlikely.numel() > 0:
            unlikely = process_unlikely(unlikely=unlikely, start_idx=i * batch_size)
            unlikely_lstgs = unlikely_lstgs + unlikely

    if len(unlikely_lstgs) > 0:
        unlikely_lstgs = x_lstg.index.values[unlikely_lstgs]
        x_lstg = x_lstg.drop(index=unlikely_lstgs)
        lookup = lookup.drop(index=unlikely_lstgs)
    frac_dropped = 100 * len(unlikely_lstgs) / org_length
    print('Dropped: {} listings ({:0.2f}%)'.format(len(unlikely_lstgs),
                                                   frac_dropped))
    return x_lstg, lookup
