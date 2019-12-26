import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import load_frames, sort_by_turns, save_files


def get_y_arrival(lstg_start, lstg_end, thread_start):
    # time_stamps
    diff = pd.to_timedelta(thread_start - lstg_start, unit='s')
    end = pd.to_timedelta(lstg_end - lstg_start, unit='s')
    # convert to intervals
    diff = (diff.dt.total_seconds() // INTERVAL['arrival']).astype('int64')
    end = (end.dt.total_seconds() // INTERVAL['arrival']).astype('int64')
    # count of arrivals by interval
    arrivals = diff.rename('period').to_frame().assign(count=1).groupby(
        ['lstg', 'period']).sum().squeeze().astype('int8')
    # error checking
    assert diff.max() < INTERVAL_COUNTS['arrival']
    assert end.max() < INTERVAL_COUNTS['arrival']
    # initialize output dataframe
    N = np.max(end)+1
    df = pd.DataFrame(0, index=end.index, dtype='int8',
        columns=range(N))
    # fill in arrivals and censored times
    for i in range(N):
        value = arrivals.xs(i, level='period').reindex(
            index=end.index, fill_value=0)
        value -= (end < i).astype('int8')
        df[i] = value
    # sort by turns and return
    return sort_by_turns(df)


def get_arrival_time_feats(lstg_start, tf):
    # add period to tf_arrival
    tf = tf.reset_index('clock')
    lstg_start = lstg_start.reindex(index=tf.index)
    tf['period'] = (tf.clock - lstg_start) // INTERVAL['arrival']
    tf = tf.drop('clock', axis=1)
    # increment period by 1; time feats are up to t-1
    tf['period'] += 1
    # drop periods beyond censoring threshold
    tf = tf[tf.period < INTERVAL_COUNTS['arrival']]
    # sum count features by period and return
    return tf.groupby(['lstg', 'period']).sum()


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # function to load file
    load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

    # outcome
    lstg_start = load_file('lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    thread_start = load_file('clock').xs(1, level='index')
    y = get_y_arrival(lstg_start, lstg_end, thread_start)
    idx = y.index

    # initialize dictionary of input features
    x = load_file('x_lstg')
    x = {k: v.reindex(index=idx) for k, v in x.items()}

    # index of first x_clock for each y
    idx_clock = lstg_start.reindex(index=idx)

    # time features
    tf_arrival = load_file('tf_arrival')
    tf = get_arrival_time_feats(lstg_start, tf_arrival).reindex(
        index=idx, level='lstg')

    return {'y': y.astype('int8', copy=False),
            'x': {k: v.astype('float32', copy=False) for k, v in x.items()}, 
            'idx_clock': idx_clock.astype('int64', copy=False),
            'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'arrival')
    