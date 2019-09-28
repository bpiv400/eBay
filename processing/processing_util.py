import pickle, numpy as np, pandas as pd
from constants import *


def load_frames(name):
    # list of paths
    paths = ['%s/%s' % (CHUNKS_DIR, p) for p in os.listdir(CHUNKS_DIR)
        if os.path.isfile('%s/%s' % (CHUNKS_DIR, p)) and name in p]
    # loop and append
    df = pd.DataFrame()
    for path in sorted(paths):
        stub = pickle.load(open(path, 'rb'))
        df = df.append(stub)
        del stub
    return df


def multiply_indices(s):
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype=np.int64)
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
    return pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])


def create_obs(df, isStart, cols):
    toAppend = pd.DataFrame(index=df.index, columns=['index'] + cols)
    for c in ['accept', 'message', 'bin']:
        if c in cols:
            toAppend[c] = False
    if isStart:
        toAppend.loc[:, 'reject'] = False
        toAppend.loc[:, 'index'] = 0
        if 'censored' in cols:
            toAppend.loc[:, 'censored'] = False
        toAppend.loc[:, 'price'] = df.start_price
        toAppend.loc[:, 'clock'] = df.start_time
    else:
        toAppend.loc[:, 'reject'] = True
        toAppend.loc[:, 'index'] = 1
        if 'censored' in cols:
            toAppend.loc[:, 'censored'] = True
        toAppend.loc[:, 'price'] = np.nan
        toAppend.loc[:, 'clock'] = df.end_time
    return toAppend.set_index('index', append=True)


def expand_index(df, levels):
    df.set_index(levels, append=True, inplace=True)
    idxcols = levels + ['lstg', 'thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def add_start_end(offers, L, levels):
    # listings dataframe
    lstgs = L[levels + ['start_date', 'end_time', 'start_price']].copy()
    lstgs['thread'] = 0
    lstgs.set_index('thread', append=True, inplace=True)
    lstgs = expand_index(lstgs, levels)
    lstgs['start_time'] = lstgs.start_date * 60 * 60 * 24
    lstgs.drop('start_date', axis=1, inplace=True)
    lstgs = lstgs.join(offers['accept'].groupby('lstg').max())
    # create data frames to append to offers
    cols = list(offers.columns)
    start = create_obs(lstgs, True, cols)
    end = create_obs(lstgs[lstgs.accept != 1], False, cols)
    # append to offers
    offers = offers.append(start, sort=True).append(end, sort=True)
    # sort
    return offers.sort_index()


def init_offers(L, T, O, levels):
    offers = O.join(T['start_time'])
    for c in ['accept', 'bin', 'reject', 'censored', 'message']:
        if c in offers:
            offers[c] = offers[c].astype(np.bool)
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[levels])
    offers = expand_index(offers, levels)
    return offers


def create_events(L, T, O, levels):
    # initial offers data frame
    offers = init_offers(L, T, O, levels)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, L, levels)
    # add features for later use
    events['byr'] = events.index.isin(IDX['byr'], level='index')
    if 'title' in levels:
        events = add_feats(events, L)
    # recode byr rejects that don't end thread
    idx = events.reset_index('index')[['index']].set_index(
        'index', append=True, drop=False).squeeze()
    tofix = events.byr & events.reject & (idx < idx.groupby(
        levels + ['index', 'thread']).transform('max'))
    events.loc[tofix, 'reject'] = False
    return events


def categories_to_string(L):
    for c in ['meta', 'leaf', 'product']:
        L[c] = c[0] + L[c].astype(str)
    mask = L['product'] == 'p0'
    L.loc[mask, 'product'] = L.loc[mask, 'leaf']
    return L