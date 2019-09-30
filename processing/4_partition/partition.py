import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd
from constants import *

LSTG_VARS = ['slr', 'lstg', 'cndtn', 'start_date', 'end_time', \
    'fdbk_score', 'fdbk_pstv', 'start_price', 'photos', 'slr_lstgs', \
    'slr_bos', 'decline_price', 'accept_price', 'store', 'slr_us', 'fast']


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


def get_x_lstg(lstgs):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # initialize output dataframe with as-is features
    df = lstgs[BINARY_FEATS + COUNT_FEATS]
    # clock features
    df['start_days'] = lstgs.start_date
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
    '''
    Creates dataframe of offer and time variables.
    '''
    # vector of offers
    offers = events.price.unstack().join(lstgs.start_price)
    offers = offers.rename({'start_price': 0}, axis=1)
    offers = offers.rename_axis('index', axis=1).stack().sort_index()
    # initialize output dataframe
    df = pd.DataFrame(index=offers.index)
    # concession DEBUG
    offers = events.price.drop(0, level='thread').unstack().join(
        L.start_price)
    offers = offers.rename({'start_price': 0}, axis=1)
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


def load_frames(name):
    # path to file number x
    path = lambda x: FEATS_DIR + str(x) + '_' + name + '.gz'
    # loop and append
    df = pd.DataFrame()
    for i in range(1,N_CHUNKS+1):
        stub = load(path(i))
        df = df.append(stub)
        del stub
    return df.sort_index()


if __name__ == "__main__":
    # load dataframes
    threads = load_frames('threads')
    events = load_frames('events')
    tf_lstg = load_frames('tf_lstg')

    # listings
    ids = tf_lstg.index
    lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv', index_col='lstg',
            usecols=LSTG_VARS).reindex(index=ids)

    # delay features
    print('Creating delay features')
    z = {}
    z['start'] = events.clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype(np.int64)
    for k, v in INTERVAL.items():
        print('\t%s' % k)
        z[k] = get_period_time_feats(tf_lstg, z['start'], k)
    dump(z, PARTS_DIR + 'z.gz')

    # outcome for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lstgs, threads)
    dump(y_arrival, PARTS_DIR + 'y_arrival.gz')

    # role outcome variables
    print('Creating role outcome variables')
    y_slr, y_byr = get_y_seq(x_offer)
    dump(y_slr, PARTS_DIR + 'y_slr.gz')
    dump(y_byr, PARTS_DIR + 'y_byr.gz')

    # thread features to save
    print('Creating thread features')
    x_thread = threads[['byr_us', 'byr_hist']]
    dump(x_thread, PARTS_DIR + 'x_thread.gz')

    # listing features
    print('Creating listing features')
    x_lstg = get_x_lstg(lstgs)
    dump(x_lstg, PARTS_DIR + 'x_lstg.gz')

    # offer features
    print('Creating offer features')
    x_offer = get_x_offer(lstgs, events, tf_lstg)
    dump(x_offer, PARTS_DIR + 'x_offer.gz')

    #  lookup file
    lookup = lstgs.reset_index().set_index(['slr', 'lstg']).sort_index()
    lookup = lookup[['start_price', 'decline_price', 'accept_price', 'start_date']]
    dump(lookup, PARTS_DIR + 'lookup.gz')
    

    