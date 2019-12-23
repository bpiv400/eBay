import pickle, argparse
from compress_pickle import load
import pandas as pd
import torch
from constants import PARTS_DIR, HOLIDAYS, PARTITIONS


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


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str, help='partition name')
    part = parser.parse_args().part
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    return part