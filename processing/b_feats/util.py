import pickle
import numpy as np
import pandas as pd
from constants import *
from processing.b_feats.time_funcs import *


def collapse_dict(feat_dict, index_names, meta=False):
    if not meta:
        remaining = [ind for ind in index_names if ind != 'thread']
        df = pd.concat(feat_dict, names=['thread'] + remaining)
    else:
        remaining = [ind for ind in index_names if ind != 'lstg_counter']
        df = pd.concat(feat_dict, names=['lstg_counter'] + remaining)
    df = df.reorder_levels(index_names).sort_index()
    return df


def get_quantiles(df, l, featname):
    # initialize output dataframe
    converter = df[['lstg_counter']]
    converter = df.set_index('lstg_counter').reset_index('lstg', drop=False)
    # subset to 1 entry per lstg per hierarchy group
    accepts = df.reset_index()[[featname, 'lstg_counter'] + l]
    accepts = accepts.groupby(by=l + ['lstg_counter']).max()

    # sanity checking for unsold lstgs
    if 0 in df.index.get_level_values('thread'):
        unsold_count = df.xs(0, level='thread')
        if 1 in unsold_count.get_level_values('index'):
            unsold_count = len(unsold_count.xs(1, level='index').index)
            assert unsold_count == accepts.isna().sum()

    # total lstgs
    total_lstgs = df.reset_index().groupby(by=l).max()['lstg_counter']
    total_lstgs = total_lstgs.reindex(accepts.index)
    quants = dict()
    # loop over quantiles
    for n in range(total_lstgs.max()):
        cut = accepts.loc[total_lstgs >= n].drop(n, level='lstg_counter')
        rel_groups = cut.index.droplevel('lstg_counter')
        cut = cut.groupby(by=l)
        partial = pd.DataFrame(index=rel_groups)
        for q in QUANTILES:
            tfname = '_'.join([l[-1], featname, str(int(100 * q))])
            partial[tfname] = cut.quantile(quantile=q,
                                           interpolation='lower').squeeze().fillna(0)
        partial = pd.concat([partial], keys=[n], names='lstg_counter')
        assert not partial.isna().any().any()
        quants[n] = partial
    # combine
    output = collapse_dict(quants, l + ['lstg_counter'], meta=True)
    assert output.index.is_unique
    output = output.join(converter)
    output = output.reset_index('lstg_counter', drop=True).set_index('lstg')
    return output


def get_cat_time_feats(events, levels):
    # initialize output dataframe
    tf = events[['clock']]
    # dataframe for variable calculations
    df = events.copy()
    df['clock'] = pd.to_datetime(df.clock, unit='s', origin=START)
    df['lstg'] = (df.index.get_level_values('index') == 0).astype(bool)
    df['thread'] = (df.index.get_level_values('index') == 1).astype(bool)
    df['slr_offer'] = ~df.byr & ~df.reject & ~df.lstg & ~df.accept
    df['byr_offer'] = df.byr & ~df.reject & ~df.accept
    df['accept_price'] = df.price[df.accept]
    df['accept_norm'] = df.price[df.accept & ~df.flag] / df.start_price
    df['lstg_id'] = df.index.get_level_values('lstg').astype(np.int64)
    df['lstg_counter'] = df['lstg_id'].groupby(by=levels).transform(
        lambda x: x.factorize()[0].astype(np.int64)
    )
    df = df.drop(columns='lstg_id')

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
        # count features grouped by current level
        ct_feats = df[['lstg', 'thread', 'slr_offer',
                       'byr_offer', 'accept']].groupby(by=l).sum()
        ctl_feats = df[['lstg', 'thread', 'slr_offer',
                       'byr_offer', 'accept']].groupby(by=l + ['lstg']).sum()
        ct_feats = ct_feats - ctl_feats
        ct_feats = ct_feats.drop('clock', axis=1)
        ct_feats = ct_feats.rename(lambda x:'_'.join([l[-1], x]) + 's', axis=1)
        ct_feats = ct_feats.astype(np.int64)
        tf = tf.join(ct_feats)

        # quantiles of (normalized) accept price over 30-day window
        quants = tf.join(get_quantiles(df, l, 'accept_norm'))
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
        tf[cols] = tf[['clock'] + cols].groupby(
            by=l + ['clock']).transform('last')
    # collapse to lstg
    tf = tf.xs(0, level='index').reset_index(levels + ['thread'], drop=True).drop('clock', axis=1)
    return tf.sort_index()


def create_obs(df, isStart, cols):
    toAppend = pd.DataFrame(index=df.index, columns=['index'] + cols)
    for c in ['accept', 'message']:
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
    for c in ['accept', 'reject', 'censored', 'message']:
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
    events = events.join(L[['flag', 'start_price']])
    return events
