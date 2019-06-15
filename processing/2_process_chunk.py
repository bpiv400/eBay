
"""
For each chunk of data, create simulator and RL inputs.
"""

import sys
sys.path.append('../')
import argparse, pickle
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd
from constants import *
from time_feats import *
from utils import *


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


def diff_time_feats(tf):
    # difference time features
    dtypes = {c: tf[c].dtype for c in tf.columns}
    diff = tf.groupby(['lstg', 'thread']).diff().dropna()
    first = tf.reset_index('clock').groupby(['lstg', 'thread']).first()
    diff = diff.append(first.set_index('clock', append=True)).sort_index()
    diff = diff.astype(dtypes).rename(lambda x: x + '_diff', axis=1)
    # create dataframe of raw and differenced features
    return tf.join(diff)


def multiply_indices(s):
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype=np.int64)
    count = 0
    # outer loop: range length
    for i in range(1, max(s)+1):
        index = s.index[s == i].values
        if len(index) == 0:
            continue
        # cartesian product of existing level(s) and period
        if k == 1:
            f = lambda x: cartesian([[x], list(range(i))])
        else:
            f = lambda x: cartesian([[e] for e in x] + [list(range(i))])
        # inner loop: rows of period
        for j in range(len(index)):
            arrays[count:count+i] = f(index[j])
            count += i
    # convert to multi-index
    return pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])


def parse_delay(df):
    # drop delays of 0
    df = df[df.delay > 0]
    # convert to period in interval
    period = df.delay.rename('period')
    period.loc[period.index.isin([2, 4, 6, 7], 
        level='index')] *= MAX_DELAY['slr'] / INTERVAL['slr']
    period.loc[period.index.isin([3, 5], 
        level='index')] *= MAX_DELAY['byr'] / INTERVAL['byr']
    period = period.astype(np.int64)
    # create multi-index from number of periods
    idx = multiply_indices(period+1)
    # expand to new index and return
    arrival = ~df[['exp']].join(period).set_index(
        'period', append=True).squeeze()
    return arrival.reindex(index=idx, fill_value=False).sort_index()


def get_y_seq(x_offer):
    d = {}
    # drop index 0
    df = x_offer.drop(0, level='index')
    # variables to restrict by
    auto = df.delay == 0
    exp = df.exp
    accept = (df.con == 1).rename('accept')
    reject = (df.con == 0).rename('reject')
    iscon = ~auto & ~exp & ~accept & ~reject
    first = df.index.get_level_values('index') == 1
    last = df.index.get_level_values('index') == 7
    # apply restrictions
    d['delay'] = parse_delay(df[~first])
    d['accept'] = accept[~auto & ~exp & ~first]
    d['reject'] = reject[~auto & ~exp & ~accept & ~first & ~last]
    d['con'] = df.con[iscon]
    d['msg'] = df['msg'][iscon]
    d['round'] = df['round'][iscon]
    d['nines'] = df['nines'][iscon & ~d['round']]
    # split by byr and slr
    slr = {k: v[v.index.isin(IDX['slr'], 
        level='index')] for k, v in d.items()}
    byr = {k: v[v.index.isin(IDX['byr'], 
        level='index')] for k, v in d.items()}
    return slr, byr


def parse_days(diff, t0, t1):
    # count of arrivals by day
    days = diff.dt.days.rename('period')
    days = days[days <= MAX_DAYS].to_frame()
    days = days.assign(count=1)
    days = days.groupby(['lstg', 'period']).sum().squeeze()
    # end of listings
    T1 = int((pd.to_datetime(END) - pd.to_datetime(START)).total_seconds())
    t1.loc[t1[t1 > T1].index] = T1
    end = (pd.to_timedelta(t1 - t0, unit='s').dt.days + 1).rename('period')
    end.loc[end > MAX_DAYS] = MAX_DAYS + 1
    # create multi-index from end stamps
    idx = multiply_indices(end)
    # expand to new index and return
    return days.reindex(index=idx, fill_value=0).sort_index()


def get_y_arrival(lstgs, threads):
    d = {}
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    t1 = lstgs.end_time
    diff = pd.to_timedelta(threads.clock - t0, unit='s')
    # append arrivals to end stamps
    d['days'] = parse_days(diff, t0, t1)
    # create other outcomes
    d['loc'] = threads.byr_us.rename('loc')
    d['hist'] = threads.byr_hist.rename('hist')
    d['bin'] = threads.bin
    sec = ((diff.dt.seconds[threads.bin == 0] + 0.5) / (24 * 3600 + 1))
    d['sec'] = sec.rename('sec')
    return d


def get_x_lstg(lstgs):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # initialize output dataframe
    df = pd.DataFrame(index=lstgs.index)
    # clock features
    df['start_days'] = lstgs.start_date
    clock = pd.to_datetime(lstgs.start_date, unit='D', origin=START)
    df = df.join(extract_day_feats(clock))
    # as-is features
    tfcols = [c for c in lstgs.columns if c.startswith('tf_')]
    for z in tfcols + BINARY_FEATS + COUNT_FEATS:
        df[z] = lstgs[z]
    # slr feedback
    df.loc[df.fdbk_score.isna(), 'fdbk_score'] = 0
    df['fdbk_score'] = df.fdbk_score.astype(np.int64)
    df['fdbk_pstv'] = lstgs['fdbk_pstv'] / 100
    df.loc[df.fdbk_pstv.isna(), 'fdbk_pstv'] = 1
    df['fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df['start'] = lstgs['start_price']
    df['decline'] = lstgs['decline_price'] / lstgs['start_price']
    df['accept'] = lstgs['accept_price'] / lstgs['start_price']
    for z in ['start', 'decline', 'accept']:
        df[z + '_round'], df[z +'_nines'] = do_rounding(df[z])
    df['has_decline'] = df['decline'] > 0
    df['has_accept'] = df['accept'] < 1
    df['auto_dist'] = df['accept'] - df['decline']
    # leaf LDA scores
    lda_weights = pickle.load(open(LDA_DIR + 'weights.pkl', 'rb'))
    w = lda_weights[:, lstgs.leaf]
    for i in range(len(lda_weights)):
        df['lda' + str(i)] = w[i, :]
    # one-hot vector for meta
    for i in range(1, 35):
        if i not in META_OTHER:
            df['meta' + str(i)] = lstgs['meta'] == i
    # condition
    cndtn = lstgs['cndtn']
    df['new'] = cndtn == 1
    df['used'] = cndtn == 7
    df['refurb'] = cndtn.isin([2, 3, 4, 5, 6])
    df['wear'] = cndtn.isin([8, 9, 10, 11]) * (cndtn - 7)
    return df


def get_x_offer(lstgs, events, tf):
    '''
    Creates dataframe of offer and time variables.
    '''
    # vector of offers
    offers = events.price.unstack().join(lstgs.start_price)
    offers = offers.rename({'start_price': 0}, axis=1)
    offers = offers.rename_axis('index', axis=1).stack().sort_index()
    # initialize output dataframe
    df = pd.DataFrame(index=offers.index)
    # concession
    df['con'] = events.con.reindex(df.index, fill_value=0)
    df['norm'] = events.norm.reindex(df.index, fill_value=0)
    df['reject'] = df['con'] == 0
    df['split'] = np.abs(df['con'] - 0.5) < TOL_HALF
    # offer digits
    df['round'], df['nines'] = do_rounding(offers)
    # message
    df['msg'] = events.message.reindex(
        df.index, fill_value=0).astype(np.bool)
    # clock variable
    clock = 24 * 3600 * lstgs.start_date.rename(0).to_frame()
    clock = clock.join(events.clock.unstack())
    # seconds since last offers
    delay = pd.DataFrame(index=clock.index)
    delay[0] = 0
    for i in range(1, 8):
        delay[i] = clock[i] - clock[i-1]
        if i in [2, 4, 6, 7]: # byr has 2 days for last turn
            censored = delay[i] > MAX_DELAY['slr']
            delay.loc[censored, i] = MAX_DELAY['slr']
            delay[i] /= MAX_DELAY['slr']
        elif i in [3, 5]:   # ignore byr arrival and last turn
            censored = delay[i] > MAX_DELAY['byr']
            delay.loc[censored, i] = MAX_DELAY['byr']
            delay[i] /= MAX_DELAY['byr']
        elif i == 1:
            delay[i] /= MAX_DELAY['byr']
    df['delay'] = delay.rename_axis('index', axis=1).stack()
    df['auto'] = df.delay == 0
    df['exp'] = (df.delay == 1) | events.censored.reindex(
        df.index, fill_value=False)
    # clock features
    df['days'] = (clock.stack() // (24 * 3600)).astype(
        np.int64) - lstgs.start_date
    df['clock'] = clock.rename_axis('index', axis=1).stack().rename(
        'clock').sort_index().astype(np.int64)
    clock = pd.to_datetime(df.clock, unit='s', origin=START)
    df = df.join(extract_day_feats(clock))
    df['minutes'] = clock.dt.hour * 60 + clock.dt.minute
    # raw time-varying features
    df = df.reset_index('index').set_index('clock', append=True)
    df = pd.concat([df, tf.reindex(df.index, fill_value=0)], axis=1)
    df = df.reset_index('clock', drop=True).set_index(
        'index', append=True)
    # change in time-varying features
    dtypes = {c: tf[c].dtype for c in tf.columns}
    tfdiff = df[tf.columns].groupby(['lstg', 'thread']).diff().dropna()
    tfdiff = tfdiff.astype(dtypes).rename(lambda x: x + '_diff', axis=1)
    df = df.join(tfdiff.reindex(df.index, fill_value=0))
    return df


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
    lstgs = L[LEVELS[:5] + ['start_date', 'end_time', 'start_price']].copy()
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
    df.set_index(LEVELS[:5], append=True, inplace=True)
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
    offers = offers.join(L[LEVELS[:5]])
    offers = expand_index(offers)
    return offers


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


def configure_events(L, T, O):
    # identify multi-listings
    ismulti = get_multi_lstgs(L)
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
    # get upper-level time-valued features
    print('Creating hierarchical time features') 
    tf_hier = get_hierarchical_time_feats(events)
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
    # split off listing events
    idx = events.xs(0, level='index').reset_index(
        'thread', drop=True).index
    lstgs = pd.DataFrame(index=idx).join(L.drop('flag', axis=1)).join(
        tf_hier.rename(lambda x: 'tf_' + x, axis=1))   
    # remove lstg start observations
    events = events.drop(0, level='thread')
    # add lstg-level time-valued features
    tf = get_lstg_time_feats(events)
    # split off threads dataframe
    events = events.join(T[['byr_hist', 'byr_us']])
    threads = events[['clock', 'byr_us', 'byr_hist', 'bin']].xs(
        1, level='index')
    events = events.drop(['byr_us', 'byr_hist', 'bin'], axis=1)
    # exclude current thread from byr_hist
    threads['byr_hist'] -= (1-threads.bin) 
    return lstgs, events, threads, tf


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    chunk = pickle.load(open(CHUNKS_DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # feature dataframes
    print('Creating events') 
    lstgs, events, threads, tf = configure_events(L, T, O)

    # input features
    print('Creating input features')
    x = {}
    x['offer'] = get_x_offer(lstgs, events, tf)
    x['thread'] = threads[['byr_us', 'byr_hist']]
    x['lstg'] = get_x_lstg(lstgs)

    # outcome variables
    print('Creating outcome variables')
    y = {}
    y['arrival'] = get_y_arrival(lstgs, threads)
    y['slr'], y['byr'] = get_y_seq(x['offer'])

    # differenced time features
    print('Differencing time features')
    z = {}
    z['start'] = events.clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype(np.int64)
    for k, v in INTERVAL.items():
        print('\t%s' % k)
        z[k] = get_period_time_feats(tf, z['start'], k)

    # write simulator chunk
    print("Writing chunk")
    chunk = {'y': y, 'x': x, 'z': z}
    pickle.dump(chunk, open(CHUNKS_DIR + '%d_out.pkl' % num, 'wb'))