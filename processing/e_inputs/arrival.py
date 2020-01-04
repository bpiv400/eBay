import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import input_partition, \
    load_frames, save_files, load_file
from processing.processing_consts import *


def get_periods(lstg_start, lstg_end):
    print('Counting periods')

    # intervals in lstg
    periods = (lstg_end - lstg_start) // INTERVAL['arrival']

    # error checking
    assert periods.max() < INTERVAL_COUNTS['arrival']

    # minimum number of periods is 1
    periods += 1

    return periods


def get_y(lstg_start, thread_start):
    print('Counting arrivals')
    
    # intervals until thread
    thread_periods = (thread_start - lstg_start) // INTERVAL['arrival']

    # error checking
    assert thread_periods.max() < INTERVAL_COUNTS['arrival']

    # count of arrivals by interval
    y = thread_periods.rename('period').to_frame().assign(
        count=1).groupby(['lstg', 'period']).sum().squeeze()

    return y


def get_tf(lstg_start, tf, periods):
    print('Collapsing time features')

    # convert clock into period
    tf['period'] = (tf.clock - lstg_start.reindex(index=tf.index)) 
    tf['period'] //= INTERVAL['arrival']
    tf = tf.drop('clock', axis=1)

    # error checking
    assert tf.period.max() < INTERVAL_COUNTS['arrival']

    # increment period by 1; time feats are up to t-1
    tf['period'] += 1

    # drop periods beyond censoring threshold
    tf = tf[tf.period < INTERVAL_COUNTS['arrival']]

    # collapse features by period
    tf = tf.groupby(['lstg', 'period']).sum()

    return tf


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # number of periods
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    periods = get_periods(lstg_start, lstg_end)

    # listing features
    x = load_file(part, 'x_lstg')
    x = {k: v.astype('float32') for k, v in x.items()}

    # arrival counts
    thread_start = load_file(part, 'clock').xs(1, level='index')
    y = get_y(lstg_start, thread_start)

    # time features
    tf = get_tf(lstg_start, load_file(part, 'tf_arrival'), periods)

    return {'periods': periods, 'y': y, 'x': x,
            'seconds': lstg_start, 'tf': tf}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    print('Saving files')
    save_files(d, part, 'arrival')
    