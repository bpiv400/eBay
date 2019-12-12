import sys, os, re
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from collections import OrderedDict
from constants import *

# data types for csv read
OTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'index': 'uint8',
          'clock': 'int64', 
          'price': 'float64', 
          'accept': bool,
          'reject': bool,
          'censored': bool,
          'message': bool}

TTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'byr': 'int64',
          'byr_hist': 'int64',
          'bin': bool,
          'byr_us': bool}

LTYPES = {'lstg': 'int64',
          'slr': 'int64',
          'meta': 'uint8',
          'cat': str,
          'cndtn': 'uint8',
          'start_date': 'uint16',
          'end_time': 'int64',
          'fdbk_score': 'int64',
          'fdbk_pstv': 'int64',
          'start_price': 'float64',
          'photos': 'uint8',
          'slr_lstgs': 'int64',
          'slr_bos': 'int64',
          'decline_price': 'float64',
          'accept_price': 'float64',
          'store': bool,
          'slr_us': bool,
          'flag': bool,
          'fast': bool}


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


# sorts y by the number of non-missing (i.e., -1) values in each row
def sort_by_turns(y):
    '''
    y: dataframe in which -1 entries should be ignored.
    '''
    # number of turns
    turns = (y > -1).sum(axis=1)
    # remove rows with 0 turns
    turns = turns[turns > 0]
    # sort by number of turns, descending
    turns = turns.sort_values(ascending=False, kind='mergesort')
    # sort y by turns
    return y.reindex(index=turns.index)


# returns dictionary of feature names for each 'x' dataframe in d
def get_featnames(d):
    '''
    d: dictionary with dataframes.
    '''

    # initialize with components of x
    featnames = {}
    featnames['x'] = OrderedDict()
    for k, v in d['x'].items():
        featnames['x'][k] = list(v.columns)

    # for arrival and delay models
    if 'x_clock' in d:
        featnames['x_time'] = \
            list(d['x_clock'].columns) + list(d['tf'].columns) + ['duration']

        # for delay models
        if 'remaining' in d:
            featnames['x_time'] += ['remaining']

    return featnames


# returns dictionary of input sizes
def get_sizes(d, model):
    '''
    d: dictionary with dataframes.
    model: one of 'arrival', 'hist', 'delay_slr', 'delay_byr', 'con_byr', 
        'con_slr', 'msg_byr', 'msg_slr'
    '''
    # number of observations
    sizes = {'N': len(d['y'].index)}

    # count components of x
    sizes['x'] = OrderedDict()
    for k, v in d['x'].items():
        sizes['x'][k] = len(v.columns)

    # arrival and delay models
    if 'x_clock' in d:
        sizes['steps'] = len(d['y'].columns)
        sizes['x_time'] = len(d['x_clock'].columns) + len(d['tf'].columns) + 1

        # delay models
        if 'remaining' in d:
            sizes['x_time'] += 1

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
    if len(np.shape(d['y'])) > 1:
        turns = np.sum(d['y'] > -1, axis=1)
        return [np.nonzero(turns == n)[0] for n in np.unique(turns)]

    # for feed-forward nets, create single vector of indices
    return [np.array(range(np.shape(d['x']['lstg'])[0]))]


# converts dictionary of dataframes to dictionary of numpy arrays
def convert_to_numpy(d):
    '''
    d: dictionary with dataframes.
    '''

    # loop through x
    for k, v in d['x'].items():
        d['x'][k] = v.astype('float32', copy=False).to_numpy()

    # convert time features to dictionary
    if 'tf' in d:
        tf_dict = {}
        for i in range(len(d['y'].index)):
            try:
                tf_dict[i] = d['tf'].xs(
                    d['y'].index[i], level='lstg')
            except:
                continue
        d['tf'] = tf_dict

    # convert y and x_fixed to numpy directly
    for k in ['y', 'x_clock', 'idx_clock', 'remaining']:
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

    # randomly select N_SMALL indices
    v = np.arange(np.shape(d['y'])[0])
    np.random.shuffle(v)
    idx = v[:N_SMALL]

    # sort indices by turns
    y = d['y'][idx]
    if len(np.shape(y)) > 1:
        turns = np.sum(y > -1, axis=1)
        idx = idx[np.argsort(-turns)]

    # directly subsample
    for k in ['y', 'idx_clock', 'remaining']:
        if k in d:
            small[k] = d[k][idx]

    # loop through x
    small['x'] = {}
    for k, v in d['x'].items():
        small['x'][k] = d['x'][k][idx]

    # x_clock in full
    if 'x_clock' in d:
        small['x_clock'] = d['x_clock']

    # time features dictionary
    if 'tf' in d:
        small['tf'] = {}
        for i in idx:
            if i in d['tf']:
                small['tf'][i] = d['tf'][i]

    # get groups for sampling
    small['groups'] = create_groups(small)

    return small


# loads x_lstg and splits into dictionary of components
def init_x(getPath, idx):
    # read in x_lstg
    df = load(getPath(['x', 'lstg']))

    # reindex
    if len(idx.names) == 1:
        df = df.reindex(index=idx)
    else:
        df = df.reindex(index=idx, level='lstg')

    # wrapper to split columns based on function
    getCols = lambda f: [c for c in df.columns if f(c)]

    # initialize dictionary of input features
    x = {}

    # features of listing
    x['lstg'] = df[['store', 'fast', 'start_price_pctile', 'start_years', \
                    'photos', 'has_photos', 'auto_decline', 'auto_accept', \
                    'start_is_round', 'start_is_nines', 'decline_is_round', \
                    'decline_is_nines', 'accept_is_round', 'accept_is_nines', \
                    'has_decline', 'has_accept', 'fdbk_pstv', 'fdbk_score', \
                    'fdbk_100', 'slr_bos_total', 'slr_lstgs_total', \
                    'new', 'used', 'refurb', 'wear']]

    # word2vec features, separately by role
    x['w2v_byr'] = df[getCols(lambda c: re.match(r'^byr[0-9]', c))]
    x['w2v_slr'] = df[getCols(lambda c: re.match(r'^slr[0-9]', c))]

    # slr features
    x['slr'] = df[getCols(lambda c: c.startswith('slr_') if c not in x['lstg'])]

    # features of category
    x['cat'] = df[getCols(lambda c: c.startswith('cat_'))]

    # features of category-condition
    x['cndtn'] = df[getCols(lambda c: c.startswith('cndtn_'))]

    return x