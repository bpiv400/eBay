import sys, os
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *


# split into chunks and save
def chunk(group, L, T, O):
    S = L[group].reset_index().set_index(group).squeeze()
    counts = S.groupby(S.index.name).count()
    groups = []
    total = 0
    num = 1
    for i in range(len(counts)):
        groups.append(counts.index[i])
        total += counts.iloc[i]
        if (i == len(counts)-1) or (total >= CUTOFF):
            # find correspinding listings
            idx = S.loc[groups]
            # create chunks
            L_i = L.reindex(index=idx)
            T_i = T.reindex(index=idx, level='lstg')
            O_i = O.reindex(index=idx, level='lstg')
            # save
            chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
            path = CHUNKS_DIR + '%s%d.gz' % (group, num)
            dump(chunk, path)
            # reinitialize
            groups = []
            total = 0
            # increment
            num += 1


# loads processed chunk files
def load_frames(name):
    '''
    name: one of 'events', 'tf_lstg', 'tf_slr'.
    '''
    # path to file number x
    path = lambda num: FEATS_DIR + '%s_%s.gz' % (num, name)
    # loop and append
    output = []
    n = len([f for f in os.listdir(FEATS_DIR) if name in f])
    for i in range(1,n+1):
        output.append(load(path(i)))
    output = pd.concat(output).sort_index()
    return output


# returns partition indices and path to file function
def get_partition(part):
    '''
    part: one of 'train_models', 'train_rl', 'test'
    '''
    partitions = load(PARTS_DIR + 'partitions.pkl')
    idx = partitions[part]
    path = lambda name: PARTS_DIR + '%s/%s.gz' % (part, name)
    return idx, path


# concatenate x_lstg, x_w2v, x_slr, x_cat
def cat_x_lstg(path):
    '''
    path: a function that takes a list of strings to be concatenated with
        underscores and returns the path to the file. Example:
        - path(['x', 'lstg']) might return 
            'data/inputs/partitions/train_models/x_lstg.gz'
    '''
    l = [load(path(['x', x])) for x in ['lstg', 'w2v', 'slr', 'cat']]
    return pd.concat(l, axis=1)


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
def extract_clock_feats(clock):
    '''
    clock: pandas series of timestamps.
    '''
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    df['minute_of_day'] = (clock.dt.hour * 60 + clock.dt.minute) / (24 * 60)
    return df


def create_x_clock():
    N = pd.to_timedelta(
        pd.to_datetime('2016-12-31 23:59:59') - pd.to_datetime(START))
    N = int((N.total_seconds()+1) / 60) # total number of minutes
    minute = pd.to_datetime(range(N), unit='m', origin=START)
    minute = pd.Series(minute, name='clock')
    return extract_clock_feats(minute).join(minute).set_index('clock')


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
            list(d['x_clock'].columns) + list(d['tf'].columns) + ['duration']
    if 'remaining' in d:
        featnames['x_time'] += ['remaining']
    if 'x_time' in d:
        featnames['x_time'] = list(d['x_time'].columns)
    return featnames


# returns dictionary of input sizes
def get_sizes(d, model):
    '''
    d: dictionary with dataframes.
    model: one of 'arrival', 'hist', 'delay_slr', 'delay_byr', 'con_byr', 
        'con_slr', 'msg_byr', 'msg_slr'
    '''
    sizes = {'N': len(d['y'].index), 
        'fixed': len(d['x_fixed'].columns)}
    if 'x_clock' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['time'] = len(d['x_clock'].columns) + len(d['tf'].columns) + 1
    if 'remaining' in d:
        sizes['time'] += 1
    if 'x_time' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['time'] = len(d['x_time'].columns)
    # output size is based on model
    if model == 'hist':
        sizes['out'] = HIST_QUANTILES
    elif 'con' in model:
        sizes['out'] = 101
    else:
        sizes['out'] = 1
    return sizes


# helper function to construct groups of equal number of turns
def create_groups(d):
    '''
    d: dictionary with numpy arrays.
    '''

    # save list of arrays of indices with same number of turns
    if 'turns' in d:
        return [np.nonzero(d['turns'] == n)[0] \
                        for n in np.unique(d['turns'])]

    # for feed-forward nets, create single vector of indices
    return [np.array(range(np.shape(d['x_fixed'])[0]))]


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
    for k in ['y', 'turns', 'x_fixed', 'x_clock', 'idx_clock', 'remaining']:
        if k in d:
            d[k] = d[k].to_numpy()

    # get groups for sampling
    d['groups'] = create_groups(d)

    return d


# restricts data to first N_SMALL observations
def create_small(d):
    '''
    d: dictionary with numpy arrays.
    '''
    
    small = {}

    # x_clock in full
    if 'x_clock' in d:
        small['x_clock'] = d['x_clock']

    # first index
    for k in ['y', 'turns', 'x_fixed', 'idx_clock', 'remaining', 'x_time']:
        if k in d:
            small[k] = d[k][:N_SMALL]

    # time features dictionary
    if 'tf' in d:
        small['tf'] = {}
        for i in range(N_SMALL):
            if i in d['tf']:
                small['tf'][i] = d['tf'][i]

    # get groups for sampling
    small['groups'] = create_groups(small)

    return small
