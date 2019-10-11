import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *

sys.path.append('repo/processing/')
from processing_utils import *


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


def parse_days(diff, t0, t1):
    # count of arrivals by day
    days = diff.dt.days.rename('period').to_frame().assign(count=1)
    days = days.groupby(['lstg', 'period']).sum().squeeze().astype(np.uint8)
    # end of listings
    T1 = int((pd.to_datetime(END) - pd.to_datetime(START)).total_seconds())
    t1[t1 > T1] = T1
    end = (pd.to_timedelta(t1 - t0, unit='s').dt.days).rename('period')
    # create multi-index from end stamps
    idx = multiply_indices(end+1)
    # expand to new index and return
    return days.reindex(index=idx, fill_value=0).sort_index()


def get_y_arrival(lstgs, threads):
    d = {}
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    t1 = lstgs.end_time
    diff = pd.to_timedelta(threads.start_time - t0, unit='s')
    # append arrivals to end stamps
    d['days'] = parse_days(diff, t0, t1)
    # create other outcomes
    d['loc'] = threads.byr_us.rename('loc')
    d['hist'] = threads.byr_hist.rename('hist')
    d['bin'] = threads.bin
    sec = ((diff.dt.seconds[threads.bin == 0] + 0.5) / (24 * 3600 + 1))
    d['sec'] = sec.rename('sec')
    return d


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


def parse_delay(df):
    # drop delays of 0
    df = df[df.delay > 0]
    # convert to period in interval
    period = df.delay.rename('period')
    period.loc[period.index.isin([2, 4, 6], 
        level='index')] *= INTERVAL_COUNTS['slr']
    period.loc[period.index.isin([3, 5], 
        level='index')] *= INTERVAL_COUNTS['byr']
    period.loc[period.index.isin([7], 
        level='index')] *= INTERVAL_COUNTS['byr_7']
    period = period.astype(np.uint8)
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


def get_x_lstg(lstgs):
    # initialize output dataframe with as-is features
    df = lstgs[BINARY_FEATS + COUNT_FEATS + ['start_date']]
    # clock features
    clock = pd.to_datetime(lstgs.start_date, unit='D', origin=START)
    df = df.join(extract_day_feats(clock))
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
    # condition
    cndtn = lstgs['cndtn']
    df['new'] = cndtn == 1
    df['used'] = cndtn == 7
    df['refurb'] = cndtn.isin([2, 3, 4, 5, 6])
    df['wear'] = cndtn.isin([8, 9, 10, 11]) * (cndtn - 7)
    return df


def get_x_offer(lstgs, events, tf):
    # vector of offers
    offers = events.price.unstack().join(lstgs.start_price)
    offers = offers.rename({'start_price': 0}, axis=1).rename_axis(
        'index', axis=1)
    # initialize output dataframe
    df = pd.DataFrame(index=offers.stack().index).sort_index()
    # concession
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    df['con'] = con.stack()
    df['reject'] = df['con'] == 0
    df['split'] = np.abs(df['con'] - 0.5) < TOL_HALF
    # total concession
    df['norm'] = events['price'] / lstgs['start_price']
    mask = events.index.isin(IDX['slr'], level='index')
    df.loc[mask, 'norm'] = 1 - df.loc[mask, 'norm']
    df.loc[df.norm.isna(), 'norm'] = 0
    # offer digits
    df['round'], df['nines'] = do_rounding(offers.stack())
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


def get_w2v(lstgs, role):
    # read in vectors
    w2v = pd.read_csv(W2V_PATH(role), index_col=0)
    # hierarchical join
    df = pd.DataFrame(np.nan, index=lstgs.index, columns=w2v.columns)
    for level in ['product', 'leaf', 'meta']:
        mask = np.isnan(df[role + '0'])
        idx = mask[mask].index
        cat = lstgs[level].rename('category').reindex(index=idx).to_frame()
        df[mask] = cat.join(w2v, on='category').drop('category', axis=1)
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # partition
    partitions = load(PARTS_DIR + 'partitions.gz')
    part = list(partitions.keys())[num-1]
    idx = partitions[part]
    path = lambda name: PARTS_DIR + part + '/' + name + '.gz'

    # load data and 
    lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv').drop(
        ['title', 'flag'], axis=1).set_index('lstg').reindex(index=idx)
    threads = pd.read_csv(CLEAN_DIR + 'threads.csv').set_index(
        ['lstg', 'thread']).reindex(index=idx, level='lstg')
    events = load_frames('events').reindex(index=idx, level='lstg')
    tf_lstg = load_frames('tf_lstg').reindex(index=idx, level='lstg')

    # lookup file
    lookup = lstgs[['meta', 'start_date', \
        'start_price', 'decline_price', 'accept_price']]
    dump(lookup, path('lookup'))

    # word2vec
    lstgs = categories_to_string(lstgs)
    w2v = get_w2v(lstgs, 'slr').join(get_w2v(lstgs, 'byr'))
    dump(w2v, path('x_w2v'))
    lstgs = lstgs.drop(['slr', 'meta', 'leaf', 'product'], axis=1)
    del w2v

    # delay start
    print('Creating delay features')
    z_start = events.clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype(np.int64)
    dump(z_start, path('z_start'))

    # delay role
    for model in ['slr', 'byr']:
        z = get_period_time_feats(tf_lstg, z_start, model)
        dump(z, path('z_' + model))
    del z, z_start

    # offer features
    print('Creating offer features')
    x_offer = get_x_offer(lstgs, events, tf_lstg)
    dump(x_offer, path('x_offer'))
    del tf_lstg, events

    # role outcome variables
    print('Creating role outcome variables')
    y = {}
    y['slr'], y['byr'] = get_y_seq(x_offer)
    for model in ['slr', 'byr']:
        for k, v in y[model].items():
            dump(v, path('_'.join(['y', model, k])))
    del x_offer, y

    # thread features to save
    print('Creating thread features')
    dump(threads[['byr_us', 'byr_hist']], path('x_thread'))

    # listing features
    print('Creating listing features')
    x_lstg = get_x_lstg(lstgs)
    dump(x_lstg, path('x_lstg'))
    del x_lstg

    # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lstgs, threads)
    for k, v in y_arrival.items():
        dump(v, path('y_arrival_' + k))