import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import load_frames, get_partition


def get_delay_time_feats(tf, start_time, role):
    # subset by role
    tf = tf[tf.index.isin(IDX[role], level='index')]
    # add period to tf_arrival
    tf = tf.reset_index('clock')
    tf = tf.join(start_time)
    tf['period'] = (tf.clock - tf.start_time) // INTERVAL[role]
    tf = tf.drop(['clock', 'start_time'], axis=1)
    # increment period by 1; time feats are up to t-1
    tf['period'] += 1
    # drop periods beyond censoring threshold
    tf = tf[tf.period < INTERVAL_COUNTS[role]]
    if role == 'byr':
        tf = tf[~tf.index.isin([7], level='index') | \
                (tf.period < INTERVAL_COUNTS['byr_7'])]
    # sum count features by period and return
    return tf.groupby(['lstg', 'thread', 'index', 'period']).sum()


def get_y_delay(delay, censored, role):
    # restrict to role indices
    s = delay[delay.index.isin(IDX[role], level='index')]
    c = censored.reindex(index=s.index)
    # expirations
    exp = s >= MAX_DELAY[role]
    if role == 'byr':
        exp.loc[exp.index.isin([7], level='index')] = s >= MAX_DELAY['slr']
    # interval of offer arrivals and censoring
    arrival = (s[~exp & ~c] / INTERVAL[role]).astype('uint16').rename('arrival')
    cens = (s[~exp & c] / INTERVAL[role]).astype('uint16').rename('cens')
    # initialize output dataframe with arrivals
    df = arrival.to_frame().assign(count=1).set_index(
        'arrival', append=True).squeeze().unstack(
        fill_value=0).reindex(index=s.index, fill_value=0)
    # vector of censoring thresholds
    v = (arrival+1).append(cens, verify_integrity=True).reindex(
        s.index, fill_value=INTERVAL_COUNTS[role])
    if role == 'byr':
        mask = v.index.isin([7], level='index') & (v > INTERVAL_COUNTS['byr_7'])
        v.loc[mask] = INTERVAL_COUNTS['byr_7']
    # replace censored observations with -1
    for i in range(INTERVAL_COUNTS[role]):
        df[i] -= (i >= v).astype('int8')
    # sort by turns and return
    return sort_by_turns(df)


if __name__ == "__main__":
    # partition and corresponding indices
    part = input_partition()
    idx, path = get_partition(part)

    # load events
    events = load(CLEAN_DIR + 'offers.pkl')[['clock', 'censored']].reindex(
        index=idx, level='lstg')
    censored = events.censored

    # calculate delay
    clock = events.clock.unstack()
    delay = pd.DataFrame(index=clock.index)
    for i in range(2, 8):
        delay[i] = clock[i] - clock[i-1]
        delay.loc[delay[i] == 0, i] = np.nan  # remove auto responses
    delay = delay.rename_axis('index', axis=1).stack().astype('int64')

    # inputs for differenced time features
    tf = load_frames('tf_delay_diff').reindex(
        index=idx, level='lstg').drop('index', axis=1)
    start_time = events.clock.rename('start_time').groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')

    # calculate role-specific features and save
    for role in ['byr', 'slr']:
        dump(get_y_delay(delay, censored, role), 
            path('y_delay_' + role))
        dump(get_delay_time_feats(tf, start_time, role), 
            path('tf_delay_diff_' + role))