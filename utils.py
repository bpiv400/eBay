import pickle
from compress_pickle import load
import pandas as pd
import torch
from rlenv.env_consts import DAY
from constants import PARTS_DIR, HOLIDAYS


def extract_day_feats(clock):
    """
    Returns dataframe with US holiday and day-of-week indicators
    :param clock: pandas series of timestamps
    :return: dataframe with holiday and day of week indicators
    """
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    return df


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


def init_x(part, idx=None):
    """
    Initializes dictionary of input components
    :param part: string name of partition (e.g., 'train_rl')
    :param idx: index of lstg ids
    :return: dictionary of pandas dataframes
    """
    x = {}

    # load and reindex
    for name in ['lstg', 'w2v_byr', 'w2v_slr', 'slr', 'cat', 'cndtn']:
        # load dataframe
        df = load(PARTS_DIR + '%s/x_%s.gz' % (part, name))
        # index by idx
        if idx is not None:
            if len(idx.names) == 1:
                df = df.reindex(index=idx)
            else:
                df = df.reindex(index=idx, level='lstg')

        # put in x
        x[name] = df

    return x