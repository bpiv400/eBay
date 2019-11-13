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
            days[i] = np.minimum(days[i], MAX_DELAY['slr'])
            delay[i] = days[i] / MAX_DELAY['slr']
        elif i in [3, 5]:   # ignore byr arrival and last turn
            days[i] = np.minimum(days[i], MAX_DELAY['byr'])
            delay[i] = days[i] / MAX_DELAY['byr']
    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / (24 * 3600)
    delay = delay.rename_axis('index', axis=1).stack()
    return days, delay


# concession
def get_con(offers):
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    return con.stack()


def get_x_offer(lookup, events):
    # vector of offers
    offers = events.price.unstack().join(lookup.start_price)
    offers = offers.rename({'start_price': 0}, axis=1).rename_axis(
        'index', axis=1)
    # initialize output dataframe
    df = pd.DataFrame(index=offers.stack().index).sort_index()
    # clock variable
    clock = 24 * 3600 * lookup.start_date.rename(0).to_frame()
    clock = clock.join(events.clock.unstack())
    # delay features
    df['days'], df['delay'] = get_delay(clock)
    # clock features
    clock = clock.rename_axis('index', axis=1).stack().rename(
        'clock').astype(np.int64)
    clock = pd.to_datetime(clock, unit='s', origin=START)
    df = df.join(extract_clock_feats(clock))
    # concession
    df['con'] = get_con(offers)
    # total concession
    df['norm'] = (events['price'] / lookup['start_price']).reindex(
        index=df.index, fill_value=0)
    df.loc[df.index.isin(IDX['slr'], level='index'), 'norm'] = \
        1 - df['norm']
    # indicator for split
    df['split'] = np.abs(df['con'] - 0.5) < TOL_HALF
    # message indicator
    df['msg'] = events['message'].reindex(
        index=df.index, fill_value=False)
    # reject auto and exp are last
    df['reject'] = (df['con'] == 0) & df.index.isin(range(1, 8), level='index')
    df['auto'] = (df.delay == 0) & df.index.isin(IDX['slr'], level='index')
    df['exp'] = df.delay == 1
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
    events = load_frames('events').reset_index(
        'slr', drop=True).reindex(index=idx, level='lstg')

    # offer features
    print('x_offer')
    x_offer = get_x_offer(lookup, events)
    dump(x_offer, path('x_offer'))

    # offer timestamps
    print('clock')
    dump(events.clock, path('clock'))
 