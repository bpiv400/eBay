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
    first = df.index.get_level_values('index') == 1
    # apply restrictions
    d['delay'] = parse_delay(df[~first])
    # split by byr and slr
    slr = {k: v[v.index.isin(IDX['slr'], 
        level='index')] for k, v in d.items()}
    byr = {k: v[v.index.isin(IDX['byr'], 
        level='index')] for k, v in d.items()}
    return slr, byr


def get_x_offer(lstgs, events):
    # vector of offers
    offers = lstgs[['start_price']].join(events.price.unstack())
    offers = offers.rename({'start_price': 0}, axis=1).rename_axis(
        'index', axis=1)
    # initialize output dataframe
    df = pd.DataFrame(index=offers.stack().index)
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
    threads = load_frames('threads').reindex(index=idx, level='lstg')
    events = load_frames('events').reindex(index=idx, level='lstg')

    # offer features
    print('Creating offer features')
    x_offer = get_x_offer(lstgs, events)

    # role outcome variables
    print('Creating role outcome variables')
    y = {}
    y['slr'], y['byr'] = get_y_seq(x_offer)
    for model in ['slr', 'byr']:
        for k, v in y[model].items():
            dump(v, path('_'.join(['y', model, k])))
    del x_offer, y

    # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lstgs, threads)
    for k, v in y_arrival.items():
        dump(v, path('y_arrival_' + k))