import numpy as np
import pandas as pd
from time_funcs import *

LEVELS = ['slr', 'meta', 'leaf', 'cndtn', 'title', 'lstg']
FUNCTIONS = {'slr': [open_lstgs, open_threads],
             'meta': [open_lstgs, open_threads],
             'leaf': [open_lstgs, open_threads],
             'cndtn': [slr_min, byr_max,
                       slr_offers, byr_offers,
                       open_lstgs, open_threads],
             'title': [slr_min, byr_max,
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
                   mapper={'start_price': 'price',
                           'start_time': 'clock',
                           'end_time': 'clock'})
    df['index'] = int('end_time' in cols)
    if 'start_price' not in cols:
        df['price'] = -1
    df['accept'] = int('end_time' in cols and 'start_price' in cols)
    df['reject'] = (df['price'] == -1).astype(np.int64)
    df.set_index('index', append=True, inplace=True)
    return df


def add_start_end(offers, listings):
    # create data frames to append to offers
    start = create_obs(listings, ['start_time', 'start_price'])
    listings = listings.join(offers['accept'].groupby('lstg').max())
    bins = create_obs(listings, ['end_time', 'start_price'],
              (listings['bin'] == 1) & (listings['accept'] != 1))
    end = create_obs(listings, ['end_time'],
        (listings['bin'] == 0) & (listings['accept'] != 1))
    # append to offers
    offers = offers.append(start).append(bins).append(end)
    # put clock into index
    offers.set_index('clock', append=True, inplace=True)
    # sort
    offers.sort_values(offers.index.names, inplace=True)
    return offers


def expand_index(df):
    df.set_index(LEVELS[:5], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_listings(L):
    listings = L[LEVELS[:5] + ['start_date', 'relisted',
                               'end_time', 'start_price', 'bin']].copy()
    listings['thread'] = 0
    listings.set_index('thread', append=True, inplace=True)
    listings = expand_index(listings)
    # convert start date to seconds
    listings['start_time'] = listings['start_date'] * 60 * 60 * 24
    listings.drop('start_date', axis=1, inplace=True)
    return listings


def init_offers(L, T, O):
    offers = O[['clock', 'price', 'accept', 'reject']].copy()
    offers = offers[~offers['price'].isna()]
    offers = offers.join(T['start_time'])
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:5]])
    offers = expand_index(offers)
    return offers


def configure_offers(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # listings data frame
    listings = init_listings(L)
    # add buy it nows
    offers = add_start_end(offers, listings)
    return offers


def get_time_feats(L, T, O):
    print('Creating offer table')
    offers = configure_offers(L, T, O)
    print('Creating time features')
    time_feats = pd.DataFrame(index=offers.index)
    for i in range(len(LEVELS)):
        levels = LEVELS[: i+1]
        ordered = offers.copy().sort_values(levels + ['clock'])
        name = levels[-1]
        f = FUNCTIONS[name]
        for j in range(len(f)):
            feat = f[j](ordered.copy(), levels)
            if f[j].__name__ in ['slr_min', 'byr_max']:
                feat = feat.div(L['start_price'])
                feat = feat.reorder_levels(
                    LEVELS + ['thread', 'index', 'clock'])
            newname = '_'.join([name, f[j].__name__])
            print('\t%s' % newname)
            time_feats[newname] = feat
    # set [lstg, thread, index] as index
    time_feats.reset_index(level=LEVELS[:len(LEVELS)-1],
                           drop=True, inplace=True)
    time_feats.reset_index(level='clock', inplace=True)
    return time_feats
