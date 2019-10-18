import sys
import os
import argparse
from compress_pickle import dump, load
import numpy as np
import pandas as pd
from processing.b_feats.time_funcs import *
from processing.b_feats.util import *
from constants import *


def thread_count(subset):
    df = subset.copy()
    s = df.reset_index()['index'] == 1
    s.index = df.index
    N = s.reset_index('thread').thread.groupby('lstg').max()
    s = s.unstack('thread')
    N = N.reindex(index=s.index, level='lstg')
    count = {}
    # count and max over non-focal threads
    for n in s.columns:
        # restrict observations
        cut = s.drop(n, axis=1).loc[N >= n]
        counts = cut.sum(axis=1)
        counts = counts.groupby('lstg').cumsum()
        count[n] = counts
    # concat into series and return
    output = pd.concat(count,names=['thread'] + s.index.names).reorder_levels(
        df.index.names).sort_index()
    return output


def add_lstg_time_feats(subset, role, isOpen):
    df = subset.copy()
    df.byr = df.byr.astype(bool)
    df.reject = df.reject.astype(bool)
    df.accept = df.accept.astype(bool)
    # set closed offers to 0
    if isOpen:
        keep = open_offers(df, ['lstg', 'thread'], role)
        assert (keep.max() == 1) & (keep.min() == 0)
        df.loc[keep == 0, 'norm'] = 0.0
    else:
        keep = None
        if role == 'slr':
            df.loc[df.byr, 'norm'] = np.nan
        elif role == 'byr':
            df.loc[~df.byr, 'norm'] = np.nan
    # use max value for thread events at same clock time
    s = df.norm.groupby(df.index.names).max()

    # number of threads in each lstg
    N = s.reset_index('thread').thread.groupby('lstg').max()
    # unstack by thread and fill with last value
    s = s.unstack(level='thread').groupby('lstg').ffill()

    if role == 'byr':
        s2 = df.byr & ~df.reject
    else:
        s2 = ~df.byr & ~df.reject
    s2 = s2.unstack(level='thread')
    if isOpen:
        keep = keep.unstack(level='thread').groupby('lstg').ffill()
    N = N.reindex(index=s.index, level='lstg')
    # initialize dictionaries for later concatenation
    count = {}
    best = {}
    # count and max over non-focal threads
    for n in s.columns:
        # restrict observations
        cut = s.drop(n, axis=1).loc[N >= n]
        cut2 = s2.drop(n, axis=1).loc[N >= n]
        if isOpen:
            cut3 = keep.drop(n, axis=1).loc[N >= n]
        # number of offers
        if isOpen and role == 'slr':
            count[n] = cut3.sum(axis=1)
        elif isOpen:
            count[n] = (cut > 0).sum(axis=1)
        else:
            counts = cut2.sum(axis=1)
            counts = counts.groupby('lstg').cumsum()
            count[n] = counts
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
    # identify listings with multiple threads
    threads = ordered.reset_index().groupby('lstg')['thread'].nunique()
    check = threads > 1
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
    tf['thread_count'] = thread_count(subset)
    # error checking
    assert (tf.byr_offers >= tf.slr_offers).min()
    assert (tf.byr_offers >= tf.byr_offers_open).min()
    assert (tf.slr_best >= tf.slr_best_open).min()
    assert (tf.byr_best >= tf.byr_best_open).min()
    # sort and return
    return tf


def get_multi_lstgs(L):
    df = L[LEVELS[:-1] + ['start_date', 'end_time']].set_index(
        LEVELS[:-1], append=True).reorder_levels(LEVELS).sort_index()
    # start time
    df['start_date'] *= 24 * 3600
    df = df.rename(lambda x: x.split('_')[0], axis=1)
    # find multi-listings
    df = df.sort_values(df.index.names[:-1] + ['start'])
    maxend = df.end.groupby(df.index.names[:-1]).cummax()
    maxend = maxend.groupby(df.index.names[:-1]).shift(1)
    overlap = df.start <= maxend
    return overlap.groupby(df.index.names).max()


def clean_events(events, L):
    # identify multi-listings
    ismulti = get_multi_lstgs(L)
    # drop multi-listings
    events = events[~ismulti.reindex(index=events.index)]
    # limit index to ['lstg', 'thread', 'index']
    events = events.reset_index(LEVELS[:-1], drop=True).sort_index()
    # 30-day burn in
    events = events.join(L['start_date'])
    events = events[events.start_date >= 30].drop('start_date', axis=1)
    # drop listings in which prices have changed
    events = events[events.flag == 0].drop('flag', axis=1)
    return events


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # quit if output files already exist
    filename = lambda x: FEATS_DIR + '%d' % num + '_' + x + '.gz'
    if os.path.isfile(filename('tf_lstg')):
        print('%d: output already exists.' % num)

    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + '%d' % num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # categories to strings
    L = categories_to_string(L)

    # set levels for hierarchical time feats
    levels = LEVELS[:6]

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating hierarchical time features') 
    tf_slr = get_cat_time_feats(events, levels)

    # drop flagged lstgs
    print('Restricting observations')
    events = clean_events(events, L)

    # split off listing events
    idx = events.reset_index('thread', drop=True).xs(
        0, level='index').index
    tf_slr = tf_slr.reindex(index=idx)
    events = events.drop(0, level='thread') # remove lstg start/end obs

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    events['norm'] = events.price / events.start_price
    events.loc[~events['byr'], 'norm'] = 1 - events['norm']


    tf_lstg = get_lstg_time_feats(events)
    events = events.drop(['byr', 'norm'], axis=1)

    # save separately
    for name in ['events', 'threads', 'tf_lstg', 'tf_slr']:
        dump(globals()[name], filename(name))