import sys
from compress_pickle import load, dump
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