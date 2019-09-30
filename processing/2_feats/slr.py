import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/2_feats/')
import argparse
from compress_pickle import dump, load
import numpy as np, pandas as pd
from constants import *
from time_funcs import *
from util import *


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

    # quit if output files already exist
    filename = lambda x: FEATS_DIR + '%d' % num + '_' + x + '.gz'
    if os.path.isfile(filename('tf_slr')):
        print('%d: output already exists.' % num), exit()

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
    lstgs = pd.DataFrame(index=idx).join(
        L.drop(['meta', 'leaf', 'product', 'flag'], axis=1))
    events = events.drop(0, level='thread') # remove lstg start/end obs

    # split off threads dataframe
    events = events.join(T[['byr_hist', 'byr_us']])
    threads = events[['clock', 'byr_us', 'byr_hist', 'bin']].xs(
        1, level='index')
    events = events.drop(['byr_us', 'byr_hist', 'bin'], axis=1)

    # exclude current thread from byr_hist
    threads['byr_hist'] -= (1-threads.bin)

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    events['norm'] = events.price / events.start_price
    events.loc[~events['byr'], 'norm'] = 1 - events['norm']
    tf_lstg = get_lstg_time_feats(events)
    events = events.drop(['byr', 'norm'], axis=1)

    # save separately
    for name in ['events', 'threads', 'lstgs', 'tf_lstg', 'tf_slr']:
        dump(globals()[name], filename(name))