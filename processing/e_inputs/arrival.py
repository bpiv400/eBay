import sys, os
from compress_pickle import load
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, \
    load_frames, save_files, load_file, init_x, \
    extract_clock_feats, get_months_since_lstg
from processing.processing_consts import CLEAN_DIR, INTERVAL
from constants import MAX_DELAY, ARRIVAL_PREFIX, START, MONTH
from featnames import THREAD_COUNT, MONTHS_SINCE_LAST


def get_arrival_times(lstg_start, lstg_end, thread_start):
    # thread 0: start of listing
    s0 = lstg_start.to_frame().assign(thread=0).set_index(
        'thread', append=True).squeeze()
    # threads 1 to N: real threads
    threads = thread_start.reset_index('thread').drop(
        'clock', axis=1).squeeze().groupby('lstg').max().reindex(
        index=lstg_start.index, fill_value=0)
    # thread N+1: end of lstg
    s1 = lstg_end.to_frame().assign(thread=threads+1).set_index(
        'thread', append=True).squeeze()
    # concatenate and sort into single series
    clock = pd.concat([s0, thread_start, s1], axis=0).sort_index()
    return clock.rename('clock')


def get_interarrival_times(clock):
    # calculate interarrival times in seconds
    df = clock.unstack()
    diff = pd.DataFrame(0.0, index=df.index, 
        columns=list(range(1, max(df.columns))))
    for i in diff.columns:
        diff[i] = df[i] - df[i-1]
    diff = diff.rename_axis('thread', axis=1).stack().astype('int64')
    # indicator for whether observation is last in lstg
    thread = pd.Series(diff.index.get_level_values(level='thread'),
        index=diff.index)
    last_thread = thread.groupby('lstg').max().reindex(
        index=thread.index, level='lstg')
    censored = thread == last_thread
    # drop interarrivals after BINs
    y = diff[diff > 0]
    censored = censored.reindex(index=y.index)
    diff = diff.reindex(index=y.index)
    # replace censored interarrival times with negative seconds from end
    y.loc[censored] -= MAX_DELAY[ARRIVAL_PREFIX]
    # convert y to periods
    y //= INTERVAL[ARRIVAL_PREFIX]
    return y, diff


def get_x_thread_feats(clock, idx, lstg_start, diff):
    # seconds since START at beginning of arrival window
    seconds = clock.groupby('lstg').shift().dropna().astype(
        'int64').reindex(index=idx)

    # clock features
    clock_feats = extract_clock_feats(
        pd.to_datetime(seconds, unit='s', origin=START))

    # thread count so far
    thread_count = pd.Series(seconds.index.get_level_values(level='thread')-1,
        index=seconds.index, name=THREAD_COUNT)

    # months since lstg start
    months_since_lstg = get_months_since_lstg(lstg_start, seconds)

    # months since last arrival
    months_since_last = diff.groupby('lstg').shift().fillna(0).rename(
        MONTHS_SINCE_LAST) / MONTH

    # concatenate into dataframe
    x_thread = pd.concat(
        [clock_feats,
        months_since_lstg,
        months_since_last,
        thread_count], axis=1)

    return x_thread.astype('float32')


def process_inputs(part):
    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    thread_start = load_file(part, 'clock').xs(1, level='index')

    # arrival times
    clock = get_arrival_times(lstg_start, lstg_end, thread_start)

    # interarrival times
    y, diff = get_interarrival_times(clock)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # add thread features to x['lstg']
    x_thread = get_x_thread_feats(clock, idx, lstg_start, diff)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1) 

    return {'y': y, 'x': x}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'arrival')
    