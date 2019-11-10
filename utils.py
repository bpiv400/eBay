import argparse, os
import pickle
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np
import pandas as pd
import torch
from constants import *


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


def get_day_inds(time):
    """
    Returns 7 indicator clock features (holiday, dow0, ..., dow5)

    :param time: current time in seconds
    :return:
    """
    clock = pd.to_datetime(time, unit='s', origin=START)
    out = torch.zeros(8).float()
    out[0] = clock.isin(HOLIDAYS)
    if clock.dayofweek < 6:
        out[clock.dayofweek + 1] = 1
    return out


# returns dataframe with US holiday and day-of-week indicators
def extract_day_feats(clock):
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    return df


def get_clock_feats(time, start_days, arrival=False, delay=False):
    """
    Gets clock features as np.array given the time

    For arrival interface, gives outputs in order of ARRIVAL_CLOCK_FEATS
    For offer interface, gives outputs in order of

    Will need to add argument to include minutes for other interface

    TODO: When days is added to delay, this will need to be changed

    :param time: int giving time in seconds
    :param start_days: int giving the day on which the lstg started
    :param arrival: boolean for whether this is an arrival model
    :param delay: boolean for whether this is for a delay model
    :return: NA
    """
    start_time = pd.to_datetime(start_days, unit='d', origin=START)
    if arrival:
        out = torch.zeros(8, dtype=torch.float64)
        min_ind = None
    elif not delay:
        out = torch.zeros(9, dtype=torch.float64)
        min_ind = 8
    else:
        out = torch.zeros(8, dtype=torch.float64)
        min_ind = 7
    clock = pd.to_datetime(time, unit='s', origin=START)

    if not delay:
        focal_days = (clock - start_time).days
        out[0] = focal_days
        hol_ind = 1
    else:
        hol_ind = 0

    out[hol_ind:(hol_ind + 7)] = get_day_inds(time)
    if not arrival:
        out[min_ind] = (clock - clock.replace(minute=0, hour=0, second=0)) / dt.timedelta(minutes=1)
    return out


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))
