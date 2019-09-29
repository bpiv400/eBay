import pickle, numpy as np, pandas as pd
from constants import *
from time_funcs import *


def get_quantiles(df, l, featname):
    # initialize output dataframe
    out = pd.DataFrame(index=df.index)
    # subset columns and group by levels
    groups = df[[featname, 'clock']].groupby(by=l)
    # function to estimation quantile q
    f = lambda q: groups.apply(lambda x: x.rolling(
        '30D', on='clock').quantile(quantile=q, interpolation='lower'))
    # loop over quantiles
    for q in QUANTILES:
        tfname = '_'.join([l[-1], featname, str(int(100 * q))])
        out[tfname] = f(q).drop('clock', axis=1).squeeze().fillna(0)
    return out


def get_cat_time_feats(events, levels):
    # initialize output dataframe
    tf = events[['clock']]
    # dataframe for variable calculations
    df = events.drop(['message', 'bin'], axis=1)
    df['clock'] = pd.to_datetime(df.clock, unit='s', origin=START)
    df['lstg'] = df.index.get_level_values('index') == 0
    df['thread'] = df.index.get_level_values('index') == 1
    df['slr_offer'] = ~df.byr & ~df.reject & ~df.lstg
    df['byr_offer'] = df.byr & ~df.reject
    df['accept_norm'] = df.norm[df.accept]
    df['accept_price'] = df.price[df.accept]
    # loop over hierarchy, exlcuding lstg
    for i in range(len(levels)):
        l = levels[: i+1]
        print(l[-1])
        # sort by levels
        df = df.sort_values(l + ['clock', 'censored'])
        tf = tf.reindex(df.index)
        # open listings
        tfname = '_'.join([l[-1], 'lstgs_open'])
        tf[tfname] = open_lstgs(df, l)
        # count features over rolling 30-day window
        ct_feats = df[['lstg', 'thread', 'slr_offer', 'byr_offer', \
            'accept', 'clock']].groupby(by=l).apply(
            lambda x: x.rolling('30D', on='clock').sum())
        ct_feats = ct_feats.drop('clock', axis=1).rename(lambda x: 
            '_'.join([l[-1], x]) + 's', axis=1).astype(np.int64)
        tf = tf.join(ct_feats)
        # quantiles of (normalized) accept price over 30-day window
        tf = tf.join(get_quantiles(df, l, 'accept_price'))
        tf = tf.join(get_quantiles(df, l, 'accept_norm'))
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
        tf[cols] = tf[['clock'] + cols].groupby(
            by=l + ['clock']).transform('last')
    # collapse to lstg
    tf = tf.xs(0, level='index').reset_index(levels + ['thread'],
        drop=True).drop('clock', axis=1)
    return tf.sort_index()


def create_obs(df, isStart, cols):
    toAppend = pd.DataFrame(index=df.index, columns=['index'] + cols)
    for c in ['accept', 'message', 'bin']:
        if c in cols:
            toAppend[c] = False
    if isStart:
        toAppend.loc[:, 'reject'] = False
        toAppend.loc[:, 'index'] = 0
        toAppend.loc[:, 'censored'] = False
        toAppend.loc[:, 'price'] = df.start_price
        toAppend.loc[:, 'clock'] = df.start_time
    else:
        toAppend.loc[:, 'reject'] = True
        toAppend.loc[:, 'index'] = 1
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
    events['norm'] = events.price / L.start_price
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