"""
Generates a dataframe of time-valued features.
"""

import pandas as pd
import numpy as np
from time_funcs import *

LEVELS = ['slr', 'meta', 'leaf', 'title', 'cndtn', 'lstg']
FUNCTIONS = {'slr': [open_lstgs, open_threads],
             'meta': [open_lstgs, open_threads],
             'leaf': [open_lstgs, open_threads],
             'title': [open_lstgs, open_threads],
             'cndtn': [slr_min, byr_max,
                       slr_offers, byr_offers,
                       open_lstgs, open_threads],
             'lstg': [slr_min, byr_max,
                      slr_offers, byr_offers,
                      open_threads]}


def create_obs(listings, cols, keep=None):
    df = listings[cols].copy()
    if keep is not None:
        df = df.loc[keep]
    df = df.rename(axis=1,
        mapper={'start_price': 'price', 'start': 'clock', 'end': 'clock'})
    df['index'] = int('end' in cols)
    if 'start_price' not in cols:
        df['price'] = -1
    df['accept'] = int('end' in cols and 'start_price' in cols)
    df['reject'] = (df['price'] == -1).astype(np.int64)
    df.set_index('index', append=True, inplace=True)
    return df


def add_start_end(offers, listings):
    # create data frames to append to offers
    start = create_obs(listings, ['start', 'start_price'])
    bins = create_obs(listings, ['end', 'start_price'],
        listings['bo'] == 0)
    end = create_obs(listings, ['end'],
        (listings['bo'] != 0) & (listings['accept'] != 1))
    # append to offers
    offers = offers.append(start).append(bins).append(end)
    # put clock into index
    offers.set_index('clock', append=True, inplace=True)
    # sort
    offers.sort_values(offers.index.names, inplace=True)
    return offers


def reindex(df):
    df.set_index(LEVELS[:5], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_listings(L, offers):
    listings = L[LEVELS[:5] + ['start_date', 'end_date', 'start_price', 'bo']].copy()
    listings['thread'] = 0
    listings.set_index('thread', append=True, inplace=True)
    listings = reindex(listings)
    # convert start and end dates to seconds
    listings['end_date'] += 1
    for z in ['start', 'end']:
        listings[z + '_date'] *= 60 * 60 * 24
        listings.rename(columns={z + '_date': z}, inplace=True)
    listings['end'] -= 1
    # indicator for whether a bo was accepted
    listings = listings.join(offers['accept'].groupby('lstg').max())
    # change end time if bo was accepted
    t_accept = offers.loc[offers['accept'] == 1, 'clock']
    listings = listings.join(t_accept.groupby('lstg').min())
    listings['end'] = listings[['end', 'clock']].min(axis=1).astype(np.int64)
    listings.drop('clock', axis=1, inplace=True)
    return listings


def init_offers(L, T, O):
    offers = O[['clock', 'price', 'accept', 'reject']].copy()
    offers = offers.join(T['start_time'])
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:5]])
    offers = reindex(offers)
    return offers


def get_offers(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # listings data frame
    listings = init_listings(L, offers)
    # add buy it nows
    offers = add_start_end(offers, listings)
    # start price for normalization
    start_price = listings['start_price'].reset_index(
        level='thread', drop=True)
    return offers, start_price


def get_time_feats(L, T, O):
    print('Creating offer dataframe')
    offers, start_price = get_offers(L, T, O)
    # offers = add_continuous_lstg(offers)
    print('Creating time features')
    time_feats = pd.DataFrame(index=offers.index)
    for i in range(len(LEVELS)):
        levels = LEVELS[:i+1]
        ordered = offers.copy().sort_values(levels + ['clock'])
        name = levels[-1]
        f = FUNCTIONS[name]
        for j in range(len(f)):
            feat = f[j](ordered.copy(), levels)
            if f[j].__name__ in ['slr_min', 'byr_max']:
                feat /= start_price
            newname = '_'.join([name, f[j].__name__])
            print('\t%s' % newname)
            feat = feat.rename(newname)
            time_feats = time_feats.join(feat)

    # set [lstg, thread, index] as index
    time_feats.reset_index(level=LEVELS[:len(LEVELS)-1],
        drop=True, inplace=True)
    time_feats.reset_index(level='clock', inplace=True)
    return time_feats