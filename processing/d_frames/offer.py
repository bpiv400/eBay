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


# concession
def get_con(offers, start_price):
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / start_price
    con[2] = (offers[2] - start_price) / (offers[1] - start_price)
    for i in range(3, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    return con.stack()


def get_x_offer(lookup, events, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=events.index).sort_index()
    # delay features
    df['days'], df['delay'] = get_delay(events.clock.unstack())
    # clock features
    clock = clock.rename_axis('index', axis=1).stack().rename(
        'clock').astype(np.int64)
    clock = pd.to_datetime(clock, unit='s', origin=START)
    df = df.join(extract_clock_feats(clock))
    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))
    # concession
    df['con'] = get_con(events.price.unstack(), lookup.start_price)
    # total concession
    df['norm'] = events.price / lookup.start_price
    df.loc[df.index.isin(IDX['slr'], level='index'), 'norm'] = 1 - df.norm
    # indicator for split
    df['split'] = np.abs(0.5 - np.around(df['con'], decimals=2)) < TOL_HALF
    # message indicator
    df['msg'] = events.message
    # reject auto and exp are last
    df['reject'] = df.con == 0
    df['auto'] = (df.delay == 0) & df.index.isin(IDX['slr'], level='index')
    df['exp'] = (df.delay == 1) | events.censored
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load other data
    lookup = load(PARTS_DIR + '%s/lookup.gz' % part)
    events = load(CLEAN_DIR + 'offers.pkl').reindex(index=idx, level='lstg')
    tf = load_frames('tf_con').reindex(index=idx, level='lstg')

    # offer features
    print('x_offer')
    x_offer = get_x_offer(lookup, events, tf)
    dump(x_offer, path('x_offer'))

    # offer timestamps
    print('clock')
    dump(events.clock, path('clock'))
 