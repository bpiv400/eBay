import sys
sys.path.append('../')
from datetime import datetime as dt
import numpy as np, pandas as pd
from time_funcs import *
from constants import *


def add_lstg_time_feats(subset, role, isOpen):
    df = subset.copy()
    # set closed offers to 0
    if isOpen:
        keep = open_offers(df, ['lstg', 'thread'], role)
        assert (keep.max() == 1) & (keep.min() == 0)
        df.loc[keep == 0, 'norm'] = 0.0
    else:
        if role == 'slr':
            df.loc[df.byr, 'norm'] = np.nan
        elif role == 'byr':
            df.loc[~df.byr, 'norm'] = np.nan
    # use max value for thread events at same clock time
    s = df.norm.groupby(df.index.names).max()
    # number of threads in each lstg
    N = s.reset_index('thread').thread.groupby('lstg').max()
    # unstack by thread and fill with last value
    s = s.unstack(level='thread').groupby('lstg').transform('ffill')
    N = N.reindex(index=s.index, level='lstg')
    # initialize dictionaries for later concatenation
    count = {}
    best = {}
    # count and max over non-focal threads
    for n in s.columns:
        # restrict observations
        cut = s.drop(n, axis=1).loc[N >= n]
        # number of offers
        count[n] = (cut > 0).sum(axis=1)
        # best offer
        best[n] = cut.max(axis=1).fillna(0.0)
    # concat into series and return
    f = lambda x: pd.concat(x, 
        names=['thread'] + s.index.names).reorder_levels(
        df.index.names).sort_index()
    return f(count), f(best)
 
 
def get_lstg_time_feats(events):
    # create dataframe for variable creation
    ordered = events.sort_values(['lstg', 'clock', 'censored']).drop(
        ['message', 'price', 'censored'], axis=1)
    # identify listings with multiple, interspersed threads
    s1 = ordered.groupby('lstg').cumcount()
    s2 = ordered.sort_index().groupby('lstg').cumcount()
    check = (s1 != s2.reindex(s1.index)).groupby('lstg').max()
    subset = ordered.loc[check[check].index].reset_index(
        'index').set_index('clock', append=True)
    # add features for open offers
    tf = pd.DataFrame()
    for role in ['slr', 'byr']:
        cols = [role + c for c in ['_offers', '_best']]
        for isOpen in [False, True]:
            if isOpen:
                cols = [c + '_open' for c in cols]
            print(cols)
            tf[cols[0]], tf[cols[1]] = add_lstg_time_feats(
                subset, role, isOpen)
    # error checking
    assert (tf.byr_offers >= tf.slr_offers).min()
    assert (tf.slr_offers >= tf.slr_offers_open).min()
    assert (tf.byr_offers >= tf.byr_offers_open).min()
    assert (tf.slr_best >= tf.slr_best_open).min()
    assert (tf.byr_best >= tf.byr_best_open).min()
    # sort and return
    return tf

def get_hierarchical_time_feats(events):
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
    for i in range(len(LEVELS)-1):
        levels = LEVELS[: i+1]
        print(levels[-1])
        # sort by levels
        df = df.sort_values(levels + ['clock', 'censored'])
        tf = tf.reindex(df.index)
        # open listings
        tfname = '_'.join([levels[-1], 'lstgs_open'])
        tf[tfname] = open_lstgs(df, levels)
        # count features over rolling 30-day window
        ct_feats = df[CT_FEATS + ['clock']].groupby(by=levels).apply(
            lambda x: x.rolling('30D', on='clock').sum())
        ct_feats = ct_feats.drop('clock', axis=1).rename(lambda x: 
            '_'.join([levels[-1], x]) + 's', axis=1).astype(np.int64)
        tf = tf.join(ct_feats)
        # quantiles of (normalized) accept price over 30-day window
        if i <= 2:
            groups = df[['accept_norm', 'clock']].groupby(by=levels)
        else:
            groups = df[['accept_price', 'clock']].groupby(by=levels)
        f = lambda q: groups.apply(lambda x: x.rolling(
            '30D', on='clock').quantile(quantile=q, interpolation='lower'))
        for q in QUANTILES:
            tfname = '_'.join([levels[-1], 'accept', str(int(100 * q))])
            tf[tfname] = f(q).drop('clock', axis=1).squeeze().fillna(0)
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
        tf[cols] = tf[['clock'] + cols].groupby(
            by=levels + ['clock']).transform('last')
    # collapse to lstg
    tf = tf.xs(0, level='index').reset_index(LEVELS[:-1] + ['thread'],
        drop=True).drop('clock', axis=1)
    return tf.sort_index()
