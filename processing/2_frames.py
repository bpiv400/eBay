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


def add_feats(events, L):
    # bool for byr turn
    events['byr'] = events.index.isin(IDX['byr'], level='index')
    # total concession
    events['norm'] = events.price / L.start_price
    events.loc[~events.byr, 'norm'] = 1 - events.norm
    # concession
    offers = events.price.drop(0, level='thread').unstack().join(
        L.start_price)
    offers = offers.rename({'start_price': 0}, axis=1)
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    events['con'] = con.stack()
    return events


def create_obs(df, isStart):
    toAppend = pd.DataFrame(index=df.index, columns=['clock', 'index',
        'censored', 'price', 'accept', 'reject', 'message', 'bin'])
    for c in ['accept', 'message', 'bin']:
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


def add_start_end(offers, L):
    # listings dataframe
    lstgs = L[LEVELS[:6] + ['start_date', 'end_time', 'start_price']].copy()
    lstgs['thread'] = 0
    lstgs.set_index('thread', append=True, inplace=True)
    lstgs = expand_index(lstgs)
    lstgs['start_time'] = lstgs.start_date * 60 * 60 * 24
    lstgs.drop('start_date', axis=1, inplace=True)
    lstgs = lstgs.join(offers['accept'].groupby('lstg').max())
    # create data frames to append to offers
    start = create_obs(lstgs, True)
    end = create_obs(lstgs[lstgs.accept != 1], False)
    # append to offers
    offers = offers.append(start, sort=True).append(end, sort=True)
    # sort
    return offers.sort_index()


def expand_index(df):
    df.set_index(LEVELS[:6], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_offers(L, T, O):
    offers = O.join(T['start_time'])
    for c in ['accept', 'bin', 'reject', 'censored', 'message']:
        offers[c] = offers[c].astype(np.bool)
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:6]])
    offers = expand_index(offers)
    return offers


def create_events(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, L)
    # add features for later use
    events = add_feats(events, L)
    # recode byr rejects that don't end thread
    idx = events.reset_index('index')[['index']].set_index(
        'index', append=True, drop=False).squeeze()
    tofix = events.byr & events.reject & (idx < idx.groupby(
        LEVELS + ['thread']).transform('max'))
    events.loc[tofix, 'reject'] = False
    return events


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
	events = events.join(L['flag'])
	events = events[events.flag == 0].drop('flag', axis=1)
	return events


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    outfile = CHUNKS_DIR + '%d_frames.pkl' % num
    chunk = pickle.load(open(CHUNKS_DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # categories to strings
    for c in ['meta', 'leaf', 'product']:
        L[c] = c[0] + L[c].astype(str)
    mask = L['product'] == 'p0'
    L.loc[mask, 'product'] = L.loc[mask, 'leaf']

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O)

    # get upper-level time-valued features
    print('Creating hierarchical time features') 
    tf_hier = get_hierarchical_time_feats(events)

    # drop flagged lstgs
    print('Restricting observations')
    events = clean_events(events, L)

    # split off listing events
    idx = events.xs(0, level='index').reset_index(
        'thread', drop=True).index
    lstgs = pd.DataFrame(index=idx).join(L.drop('flag', axis=1)).join(
        tf_hier.rename(lambda x: 'tf_' + x, axis=1))   
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