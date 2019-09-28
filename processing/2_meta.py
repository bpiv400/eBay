import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
import numpy as np, pandas as pd
from constants import *
from time_funcs import *


def get_cat_time_feats(events):
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


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1
    prefix = CHUNKS_DIR + 'm%d' % num

    # load data
    print('Loading data')
    chunk = pickle.load(open(prefix + '.pkl', 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # categories to strings
    L = categories_to_string(L)

    # set levels for hierarchical time feats
    levels = LEVELS[1:4]

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating categorical time features') 
    tf_cat = get_cat_time_feats(events)

    # save
    pickle.dump(tf_cat, open(prefix + '_tf.pkl', 'wb'))
    