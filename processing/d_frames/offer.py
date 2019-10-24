import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


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


def split_by_role(s):
    byr = s[s.index.isin(IDX['byr'], level='index')]
    slr = s[s.index.isin(IDX['slr'], level='index')]
    return byr, slr


def get_y_delay(x_offer):
    # drop indices 0 and 1
    period = x_offer['delay'].drop([0, 1], level='index').rename('period')
    # remove delays of 0
    period = period[period > 0]
    # convert to period in interval
    period.loc[period.index.isin([2, 4, 6], 
        level='index')] *= INTERVAL_COUNTS['slr']
    period.loc[period.index.isin([3, 5], 
        level='index')] *= INTERVAL_COUNTS['byr']
    period.loc[period.index.isin([7], 
        level='index')] *= INTERVAL_COUNTS['byr_7']
    period = period.astype(np.uint8)
    # create multi-index from number of periods
    idx = multiply_indices(period+1)
    # expand to new index
    offer = period.to_frame().assign(offer=False).set_index(
        'period', append=True).squeeze()
    offer = offer.reindex(index=idx, fill_value=False).sort_index()
    # split by role and return
    return split_by_role(offer)


def get_y_con(x_offer):
    # drop zero delay and expired offers
    mask = (x_offer.delay > 0) & ~x_offer.exp
    s = x_offer.loc[mask, 'con']
    # split by role and return
    return split_by_role(s)


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
    df['norm'] = (events['price'] / lstgs['start_price']).reindex(
        index=df.index, fill_value=0.0)
    df.loc[df.index.isin(IDX['slr'], level='index'), 'norm'] = \
        1 - df['norm']
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
    df['hour_of_day'] = clock.dt.hour
    # raw time-varying features
    df = df.reset_index('index').set_index('clock', append=True)
    df = pd.concat([df, tf.reindex(df.index, fill_value=0).rename(
        lambda x: x + '_raw', axis=1)], axis=1)
    df = df.reset_index('clock', drop=True).set_index(
        'index', append=True)
    # change in time-varying features
    tfeats = [c for c in df.columns if c.endswith('_raw')]
    dtypes = {c: df[c].dtype for c in tfeats}
    tfdiff = df[tfeats].groupby(['lstg', 'thread']).diff().dropna()
    tfdiff = tfdiff.astype(dtypes).rename(
        lambda x: x.replace('_raw', '_diff'), axis=1)
    df = df.join(tfdiff.reindex(df.index, fill_value=0))
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load data and 
    lstgs = load(CLEAN_DIR + 'listings.gz')
    lstgs = lstgs[['start_price', 'start_date']].reindex(index=idx)
    events = load_frames('events').reindex(index=idx, level='lstg')
    tf = load_frames('tf_lstg').reindex(index=idx, level='lstg')

    # delay start
    print('z')
    z_start = events.clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype(np.int64)
    dump(z_start, path('z_start'))

    # delay role
    for role in ['slr', 'byr']:
        z = get_period_time_feats(tf, z_start, role)
        dump(z, path('z_' + role))

    # offer features
    print('x_offer')
    x_offer = get_x_offer(lstgs, events, tf)
    dump(x_offer, path('x_offer'))

    # delay outcome
    print('y_delay')
    y_delay_byr, y_delay_slr = get_y_delay(x_offer)
    dump(y_delay_byr, path('y_delay_byr'))
    dump(y_delay_slr, path('y_delay_slr'))

    # concession outcome
    print('y_con')
    y_con_byr, y_con_slr = get_y_con(x_offer)
    dump(y_con_byr, path('y_con_byr'))
    dump(y_con_slr, path('y_con_slr'))
 