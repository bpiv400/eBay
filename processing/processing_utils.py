import sys
from compress_pickle import load, dump
from sklearn.utils.extmath import cartesian
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *


# loads processed chunk files
def load_frames(name):
    '''
    name: one of 'events', 'tf_lstg', 'tf_slr'.
    '''
    # path to file number x
    path = lambda num: FEATS_DIR + '%s_%s.gz' % (num, name)
    # loop and append
    output = []
    n = CHUNKS_CAT if name == 'cat' else CHUNKS_SLR
    for i in range(1,n+1):
        output.append(load(path(i)))
    output = pd.concat(output).sort_index()
    return output


# returns partition indices and path to file function
def get_partition(part):
    '''
    part: one of 'train_models', 'train_rl', 'test'
    '''
    partitions = load(PARTS_DIR + 'partitions.gz')
    idx = partitions[part]
    path = lambda name: PARTS_DIR + '%s/%s.gz' % (part, name)
    return idx, path


# appends turn indicator variables to offer matrix
def add_turn_indicators(df):
    '''
    df: dataframe with index ['lstg', 'thread', 'index'].
    '''
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


# creates features from timestapes
def extract_clock_feats(clock, start):
    '''
    clock: pandas series of timestamps.
    '''
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    df['minute_of_day_norm'] = (clock.dt.hour * 60 + clock.dt.minute) / (24 * 60)
    return df


# count number of time steps in each observations
def get_sorted_turns(y):
    '''
    y: dataframe of outcomes
        - columns are time steps
        - missing outcomes are coded -1
    '''
    turns = (y > -1).sum(axis=1)
    return turns.sort_values(ascending=False, kind='mergesort')


# returns dictionary of feature names for each 'x' dataframe in d
def get_featnames(d):
    '''
    d: dictionary with dataframes.
    '''
    featnames = {'x_fixed': list(d['x_fixed'].columns)}
    if 'x_clock' in d:
        featnames['x_time'] = \
            list(d['x_clock'].columns) + list(d['tf'].columns)
    if 'x_time' in d:
        featnames['x_time'] = list(d['x_time'].columns)
    return featnames


# returns dictionary of input sizes
def get_sizes(d):
    '''
    d: dictionary with dataframes.
    '''
    sizes = {'N': len(d['y'].index), 
        'fixed': len(d['x_fixed'].columns)}
    if 'x_clock' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['time'] = len(d['x_clock'].columns) + len(d['tf'].columns)
    if 'x_time' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['time'] = len(d['x_time'].columns)
    return sizes


# converts dictionary of dataframes to dictionary of numpy arrays
def convert_to_numpy(d):
    '''
    d: dictionary with dataframes.
    '''

    # convert time features to dictionary
    if 'tf' in d:
        tf_dict = {}
        for i in range(len(d['turns'].index)):
            try:
                tf_dict[i] = d['tf'].xs(
                    d['turns'].index[i], level='lstg')
            except:
                continue
        d['tf'] = tf_dict

    # reshape columns of x_time and convert
    if 'x_time' in d:
        arrays = []
        for c in d['x_time'].columns:
            array = d['x_time'][c].astype('float32').unstack().reindex(
                index=d['y'].index).to_numpy()
            arrays.append(np.expand_dims(array, axis=2))
        d['x_time'] = np.concatenate(arrays, axis=2)

    # convert y and x_fixed to numpy directly
    for k in ['y', 'turns', 'x_fixed', 'idx_clock', 'x_clock']:
        if k in d:
            d[k] = d[k].to_numpy()

    # save list of arrays of indices with same number of turns
    if 'turns' in d:
        d['groups'] = [np.nonzero(d['turns'] == n)[0] \
                        for n in np.unique(d['turns'])]

    # for feed-forward nets, create single vector of indices
    else:
        d['groups'] = [np.array(range(np.shape(d['x_fixed'])[0]))]

    return d
