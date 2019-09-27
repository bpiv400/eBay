import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
import numpy as np, pandas as pd
from constants import *
from time_funcs import *


def get_period_time_feats(tf, start, model):
    # initialize output
    output = pd.DataFrame()
    # loop over indices
    for i in IDX[model]:
        if i == 1:
            continue
        df = tf.reset_index('clock')
        # count seconds from previous offer
        df['clock'] -= start.xs(i, level='index').reindex(df.index)
        df = df[~df.clock.isna()]
        df = df[df.clock >= 0]
        df['clock'] = df.clock.astype(np.int64)
        # add index
        df = df.assign(index=i).set_index('index', append=True)
        # collapse to period
        df['period'] = (df.clock - 1) // INTERVAL[model]
        df['order'] = df.groupby(df.index.names + ['period']).cumcount()
        df = df.sort_values(df.index.names + ['period', 'order'])
        df = df.groupby(df.index.names + ['period']).last().drop(
            ['clock', 'order'], axis=1)
        # reset clock to beginning of next period
        df.index.set_levels(df.index.levels[-1] + 1, 
            level='period', inplace=True)
        # appoend to output
        output = output.append(df)
    return output.sort_index()


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


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    infile = CHUNKS_DIR + '%d_frames.pkl' % num
    d = pickle.load(open(infile, 'rb'))
    events, lstgs, threads = [d[k] for k in ['events', 'lstgs', 'threads']]

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    tf = get_lstg_time_feats(events)
    
    # differenced time features
    print('Differencing time features')
    z = {}
    z['start'] = events.clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype(np.int64)
    for k, v in INTERVAL.items():
        print('\t%s' % k)
        z[k] = get_period_time_feats(tf, z['start'], k)

    # save separately
    filename = lambda x: CHUNKS_DIR + '%d_' % num + x + '.pkl'
    for name in ['events', 'lstgs', 'threads', 'tf', 'z']:
        pickle.dump(globals()[name], open(filename(name), 'wb'))