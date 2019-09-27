import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing_utils import *


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


if __name__ == "__main__":
    # load dataframes
    lstgs = load_frames('lstgs')
    events = load_frames('events')
    tf = load_frames('tf')
    z = load_frames('z')

    # delay features
    print('Creating delay features')
    pickle.dump(z, open(FRAMES_DIR + 'z.pkl', 'wb'))

    # offer features
    print('Creating offer features')
    x_offer = get_x_offer(lstgs, events, tf)
    pickle.dump(x_offer, open(FRAMES_DIR + 'x_offer.pkl', 'wb'))

    # role outcome variables
    print('Creating role outcome variables')
    y_slr, y_byr = get_y_seq(x_offer)
    pickle.dump(y_slr, open(FRAMES_DIR + 'y_slr.pkl', 'wb'))
    pickle.dump(y_byr, open(FRAMES_DIR + 'y_byr.pkl', 'wb'))

    