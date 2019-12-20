import sys, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# delay
def get_delay(clock):
    # initialize output dataframes in wide format
    days = pd.DataFrame(0., index=clock.index, columns=clock.columns)
    delay = pd.DataFrame(0., index=clock.index, columns=clock.columns)
    for i in range(2, 8):
        days[i] = clock[i] - clock[i-1]
        if i in [2, 4, 6, 7]: # byr has 2 days for last turn
            delay[i] = days[i] / MAX_DELAY['slr']
        elif i in [3, 5]:   # ignore byr arrival and last turn
            delay[i] = days[i] / MAX_DELAY['byr']
    # no delay larger than 1
    assert delay.max().max() <= 1
    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / (24 * 3600)
    delay = delay.rename_axis('index', axis=1).stack()
    return days, delay


# round concession to nearest percentage point
def round_con(con):
    '''
    con: series of unrounded concessions
    '''
    rounded = np.round(con, decimals=2)
    rounded.loc[(rounded == 1) & (con < 1)] = 0.99
    rounded.loc[(rounded == 0) & (con > 0)] = 0.01
    return rounded


# concession
def get_con(offers, start_price):
    con = pd.DataFrame(index=offers.index)
    con[1] = offers[1] / start_price
    con[2] = (offers[2] - start_price) / (offers[1] - start_price)
    for i in range(3, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    return round_con(con.rename_axis('index', axis=1).stack())


# calculate normalized concession from rounded concessions
def get_norm(con):
    con = con.unstack()
    norm = pd.DataFrame(index=con.index, columns=con.columns)
    norm[1] = con[1]
    norm[2] = con[2] * (1-norm[1])
    for i in range(3, 8):
        if i in IDX['byr']:
            norm[i] = con[i] * (1-norm[i-1]) + (1-con[i]) * norm[i-2]
        elif i in IDX['slr']:
            norm[i] = 1 - con[i] * norm[i-1] - (1-con[i]) * (1-norm[i-2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def get_x_offer(start_price, events, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=events.index).sort_index()
    # delay features
    df['days'], df['delay'] = get_delay(events.clock.unstack())
    # clock features
    clock = pd.to_datetime(events.clock, unit='s', origin=START)
    df = df.join(extract_clock_feats(clock))
    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))
    # concession
    df['con'] = get_con(events.price.unstack(), start_price)
    # total concession
    df['norm'] = get_norm(df.con)
    # indicator for split
    df['split'] = (df.con >= 0.49) & (df.con <= 0.51)
    # message indicator
    df['msg'] = events.message
    # reject auto and exp are last
    df['reject'] = df.con == 0
    df['auto'] = (df.delay == 0) & df.index.isin(IDX['slr'], level='index')
    df['exp'] = (df.delay == 1) | events.censored
    return df


if __name__ == "__main__":
    # parameter(s) from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', action='store', type=str, required=True)
    part = parser.parse_args().part
    idx, path = get_partition(part)

    # load other data
    start_price = load(PARTS_DIR + '%s/lookup.gz' % part).start_price
    events = load(CLEAN_DIR + 'offers.pkl').reindex(index=idx, level='lstg')
    tf = load_frames('tf_con').reindex(index=idx, level='lstg')

    # offer features
    print('x_offer')
    x_offer = get_x_offer(start_price, events, tf)
    dump(x_offer, path('x_offer'))

    # offer timestamps
    print('clock')
    dump(events.clock, path('clock'))
 