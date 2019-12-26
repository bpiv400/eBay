import pickle, argparse
from compress_pickle import load
import numpy as np, pandas as pd
from datetime import datetime as dt
from constants import PARTS_DIR, HOLIDAYS, PARTITIONS, START, END, MAX_DAYS


def init_date_lookup():
    start = pd.to_datetime(START)
    end = pd.to_datetime(END) + pd.to_timedelta(MAX_DAYS, unit='D')

    d = {}
    for date in pd.date_range(start.date(), end.date()): 
        row = np.zeros(8)
        row[0] = str(date.date()) in HOLIDAYS
        for i in range(6):
            row[i+1] = date.dayofweek == i
        d[(date - start).days] = row

    return d


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


def extract_clock_feats(clock):
    '''
    Creates clock features from timestamps.
    :param clock: pandas series of timestamps.
    :return: pandas dataframe of holiday and day of week indicators, and minute of day.
    '''
    df = extract_day_feats(clock)

    # add in seconds of day
    seconds_since_midnight = clock.dt.hour * 60 + clock.dt.minute + clock.dt.second
    df['second_of_day'] = (seconds_since_midnight / (24 * 3600)).astype('float32')
    return df


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


def input_partition(store_true=None):
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str, help='partition name')

    # optional boolean arguments
    if store_true is not None:
        parser.add_argument('--{}'.format(store_true), 
            action='store_true', default=False)

    # parse arguments
    args = parser.parse_args()

    # error checking
    if args.part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))

    # return tuple if multiple arguuments
    if store_true is None:
        return args.part
    else:
        return args