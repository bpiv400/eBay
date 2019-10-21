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
    for i in range(1,N_CHUNKS+1):
        output.append(load(path(i)))
    output = pd.concat(output).sort_index()
    return output


# converts meta, leaf and product to str, replaces missing product w/leaf
def categories_to_string(L):
    '''
    L: dataframe with index 'lstg' and columns 'meta', 'leaf', and 'product'
    '''
    for c in ['meta', 'leaf', 'product']:
        L[c] = np.char.add(c[0], L[c].astype(str).values)
    mask = L['product'] == 'p0'
    L.loc[mask, 'product'] = L.loc[mask, 'leaf']
    return L


# for creating arrival and delay outcomes
def multiply_indices(s):
    '''
    s: Series with index ['lstg'] or ['lstg', 'thread'] and number of periods.
    '''
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype='uint16')
    count = 0
    # outer loop: range length
    for i in range(1, max(s)+1):
        index = s.index[s == i].values
        if len(index) == 0:
            continue
        # cartesian product of existing level(s) and period
        if k == 1:
            f = lambda x: cartesian([[x], list(range(i))])
        else:
            f = lambda x: cartesian([[e] for e in x] + [list(range(i))])
        # inner loop: rows of period
        for j in range(len(index)):
            arrays[count:count+i] = f(index[j])
            count += i
    # convert to multi-index
    idx = pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])
    return idx.sortlevel(idx.names)[0]


# returns partition indices and path to file function
def get_partition(num):
    '''
    num: 1, 2 or 3; to index PARTITIONS.
    '''
    partitions = load(PARTS_DIR + 'partitions.gz')
    part = list(partitions.keys())[num-1]
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


# returns dictionary of feature names for each 'x' dataframe in d
def get_featnames(d):
    '''
    d: dictionary with dataframes.
    '''
    featnames = {'x_fixed': list(d['x_fixed'].columns)}
    if 'x_hour' in d:
        featnames['x_fixed'] += list(d['x_hour'].rename(
            lambda x: x + '_focal', axis=1).columns)
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
    if 'x_hour' in d:
        sizes['fixed'] += len(d['x_hour'].columns)
    if 'x_time' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['time'] = len(d['x_time'].columns)
    return sizes


# converts dictionary of dataframes to dictionary of numpy arrays
def convert_to_numpy(d):
    '''
    d: dictionary with dataframes.
    '''
    # reshape columns of x_time and convert
    if 'x_time' in d:
        arrays = []
        for c in d['x_time'].columns:
            array = d['x_time'][c].astype('float32').unstack().reindex(
                index=d['y'].index).to_numpy()
            arrays.append(np.expand_dims(array, axis=2))
        d['x_time'] = np.concatenate(arrays, axis=2)

    # convert y and x_fixed to numpy directly
    for k in ['y', 'x_fixed']:
        d[k] = d[k].to_numpy()

    return d
