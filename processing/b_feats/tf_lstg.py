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
    # print('thread count input')
    # print(df)
    index_names = [ind for ind in df.index.names if ind != 'index']
    s = df.reset_index()['index'] == 1
    s.index = df.index
    N = s.reset_index('thread').thread.groupby('lstg').max()
    s = s.unstack('thread')
    # print('s thread count')
    # print(s)
    N = N.reindex(index=s.index, level='lstg')
    count = {}
    # count and max over non-focal threads
    for n in s.columns:
        # restrict observations
        cut = s.drop(n, axis=1).loc[N >= n]
        # print('thread count cut')
        counts = cut.sum(axis=1)
        counts = counts.groupby('lstg').cumsum()
        counts = counts.reset_index('index', drop=True)
        counts = counts.groupby(level=counts.index.names).last()
        count[n] = counts
    # concat into series and return
    thread_ind_names = count[1].index.names
    output = pd.concat(count,names=['thread'] + thread_ind_names).reorder_levels(
        index_names).sort_index()
    return output


def add_lstg_time_feats(subset, role, isOpen):
    print('input')
    print(subset)
    subset.byr = subset.byr.astype(bool)
    subset.reject = subset.reject.astype(bool)
    subset.accept = subset.accept.astype(bool)
    index_names = [ind for ind in subset.index.names if ind != 'index']
    df = subset.copy().reset_index('index')
    df.byr = df.byr.astype(bool)
    df.reject = df.reject.astype(bool)
    df.accept = df.accept.astype(bool)
    df2 = subset.copy()
    # print('done copying')
    # set closed offers to 0
    if isOpen:
        keep = open_offers(df2, ['lstg', 'thread'], role)
        assert (keep.max() == 1) & (keep.min() == 0)
        df2.loc[keep == 0, 'norm'] = 0.0
        # keep = open_offers(df2, ['lstg', 'thread'], role)
        # print('keep')
        # print(keep)
    else:
        keep = None
        if role == 'slr':
            df2.loc[df2.byr, 'norm'] = np.nan
        elif role == 'byr':
            df2.loc[~df2.byr, 'norm'] = np.nan
    # use max value for thread events at same clock time
    s = df2.norm.groupby(df2.index.names).max()

    # number of threads in each lstg
    N = s.reset_index('thread').thread.groupby('lstg').max()
    # unstack by thread and fill with last value
    s = s.unstack(level='thread').groupby('lstg').ffill()

    if role == 'byr':
        s2 = df2.byr & ~df2.reject
        print('s2')
        print(s2)
    else:
        s2 = ~df2.byr & ~df2.reject
    print('s2 setup')
    N2 = s2.reset_index('thread').thread.groupby('lstg').max()
    print('n2 setup')
    s2 = s2.unstack(level='thread')
    if isOpen:
        keep = keep.unstack(level='thread').groupby('lstg').ffill()
    N = N.reindex(index=s.index, level='lstg')
    N2 = N2.reindex(index=s2.index, level='lstg')
    print('n2 reindex')
    # initialize dictionaries for later concatenation
    count = {}
    best = {}
    # count and max over non-focal threads
    for n in s.columns:
        # restrict observations
        cut = s.drop(n, axis=1).loc[N >= n]
        cut2 = s2.drop(n, axis=1).loc[N2 >= n]
        # print('cut the boy')
        if isOpen and role == 'slr':
            cut3 = keep.drop(n, axis=1).loc[N2 >= n]
            cut3 = cut3.sum(axis=1).reset_index('index', drop=True)
            cut3 = cut3.groupby(level=cut3.index.names).last()
            count[n] = cut3
        elif isOpen:
            counts = (cut > 0).sum(axis=1)
            counts = counts.reset_index('index', drop=True)
            counts = counts.groupby(level=counts.index.names).last()
            # print('after dups: {}'.format(counts.index.drop_duplicates(keep='last').is_unique))
            count[n] = counts  # TODO may need to change due to automatic problems
            # print('ordinary version')
            # print(count[n])
            # print(count[n].index.names)
        else:
            print('cut 2')
            print(cut2)
            counts = cut2.sum(axis=1)
            counts = counts.groupby('lstg').cumsum()
            counts = counts.reset_index('index', drop=True)
            counts = counts.groupby(level=counts.index.names).last()
            # print('after dups: {}'.format(counts.index.drop_duplicates(keep='last').is_unique))
            # print('reset the boy')
            # print(counts)
            # print('counts unique: {}'.format(counts.index.is_unique))
            # print(counts.index.names)
            count[n] = counts

        # best offer
        curr_best = cut.max(axis=1).fillna(0.0)
        curr_best = curr_best.reset_index('index', drop=True)
        curr_best = curr_best.groupby(level=curr_best.index.names).last()
        best[n] = curr_best
    # concat into series and return
    print('concat is all that left')
    f = lambda x: pd.concat(x, 
        names=['thread'] + ['lstg', 'clock']).reorder_levels(
        index_names).sort_index()
    output = f(count), f(best)
    print('done concat')
    return output


def get_lstg_time_feats(events):
    # create dataframe for variable creation
    ordered = events.sort_values(['lstg', 'clock', 'index', 'censored']).drop(
        ['message', 'price'], axis=1)
    # identify listings with multiple threads
    threads = ordered.reset_index().groupby('lstg')['thread'].nunique()
    check = threads > 1
    subset = ordered.loc[check[check].index].set_index(['clock'], append=True)
    subset = subset.reorder_levels(['lstg', 'thread', 'clock', 'index'])
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
    print(tf.index.names)
    return tf

if __name__ == "__main__":
	# parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    FEATS_DIR = 'data/feats/'
    print(FEATS_DIR + '%d_events.gz' % num)
    events = load(FEATS_DIR + '%d_events.gz' % num)

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    events['norm'] = events.price / events.start_price
    events.loc[~events['byr'], 'norm'] = 1 - events['norm']
    tf_lstg = get_lstg_time_feats(events)

    # save
    dump(tf_lstg, FEATS_DIR + '%d_tf_lstg.gz' % num)
