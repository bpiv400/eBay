import sys
from compress_pickle import load, dump
from sklearn.utils.extmath import cartesian
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *


# loads processed chunk files
def load_frames(name):
    # path to file number x
    path = lambda x: FEATS_DIR + str(x) + '_' + name + '.gz'
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
    partitions = load(PARTS_DIR + 'partitions.gz')
    part = list(partitions.keys())[num-1]
    idx = partitions[part]
    path = lambda name: PARTS_DIR + '%s/%s.gz' % (part, name)
    return idx, path
