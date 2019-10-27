import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# delay
def get_delay(clock):
    delay = pd.DataFrame(index=clock.index)
    delay[0] = 0
    delay[1] = (clock[1] - clock[0]) / MAX_DELAY['byr']
    for i in range(2, 8):
        delay[i] = clock[i] - clock[i-1]
        if i in [2, 4, 6, 7]: # byr has 2 days for last turn
            delay[i] /= MAX_DELAY['slr']
        elif i in [3, 5]:   # ignore byr arrival and last turn
            delay[i] /= MAX_DELAY['byr']
        # censor delays at MAX_DELAY
        delay.loc[delay[i] > 1, i] = 1
    return delay.rename_axis('index', axis=1).stack()


# concession
def get_con(offers):
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    return con.stack()


def get_x_offer(lstgs, events, tf):
    # vector of offers
    offers = events.price.unstack().join(lstgs.start_price)
    offers = offers.rename({'start_price': 0}, axis=1).rename_axis(
        'index', axis=1)
    # initialize output dataframe
    df = pd.DataFrame(index=offers.stack().index).sort_index()
    # concession
    df['con'] = get_con(offers)
    df['reject'] = df['con'] == 0
    df['split'] = np.abs(df['con'] - 0.5) < TOL_HALF
    # total concession
    df['norm'] = (events['price'] / lstgs['start_price']).reindex(
        index=df.index, fill_value=0.0)
    df.loc[df.index.isin(IDX['slr'], level='index'), 'norm'] = \
        1 - df['norm']
    # message indicator
    df['msg'] = events['message']
    # clock variable
    clock = 24 * 3600 * lstgs.start_date.rename(0).to_frame()
    clock = clock.join(events.clock.unstack())
    # delay features
    df['delay'] = get_delay(clock)
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

    # load data
    lstgs = load(CLEAN_DIR + 'listings.gz')
    lstgs = lstgs[['start_price', 'start_date']].reindex(index=idx)
    events = load_frames('events').reindex(index=idx, level='lstg')
    tf = load_frames('tf_lstg').reindex(index=idx, level='lstg')

    # offer features
    print('x_offer')
    x_offer = get_x_offer(lstgs, events, tf)
    dump(x_offer, path('x_offer'))
 