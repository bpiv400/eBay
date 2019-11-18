import pickle
from compress_pickle import load
import pandas as pd
import torch
from rlenv.env_consts import DAY
from constants import PARTS_DIR, HOLIDAYS, HIST_QUANTILES


# concatenates listing features into single dataframe
def cat_x_lstg(part):
    """
    part: one of 'test', 'train_models', 'train_rl'
    """
    prefix = PARTS_DIR + part + '/' + 'x_'
    x_lstg = load(prefix + 'lstg.gz')
    for s in ['slr', 'meta', 'w2v']:
        filename = prefix + s + '.gz'
        x_lstg = x_lstg.join(load(filename), rsuffix=s)
    return x_lstg


# returns dataframe with US holiday and day-of-week indicators
def extract_day_feats(clock):
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    return df


def add_out_to_sizes(model, sizes):
    if model in ['hist', 'con_byr', 'con_slr']:
        if model == 'hist':
            sizes['out'] = HIST_QUANTILES
        else:
            sizes['out'] = 101
    else:
        sizes['out'] = 1


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))
