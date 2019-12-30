import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import input_partition, load_frames, save_files
from processing.processing_consts import *


def get_periods(lstg_start, lstg_end):
    # seconds in listing
    end = pd.to_timedelta(lstg_end - lstg_start, 
        unit='s').dt.total_seconds().astype('int64')

    # convert to interval
    periods = end // INTERVAL['arrival']

    # error checking
    assert periods.max() < INTERVAL_COUNTS['arrival']

    # sorted number of turns
    periods += 1
    periods.sort_values(ascending=False, inplace=True)

    return periods


def get_y(lstg_start, thread_start, periods):
    # time_stamps
    diff = pd.to_timedelta(thread_start - lstg_start, 
        unit='s').dt.total_seconds().astype('int64')
    
    # convert to intervals
    diff_period = diff // INTERVAL['arrival']

    # error checking
    assert diff_period.max() < INTERVAL_COUNTS['arrival']

    # count of arrivals by interval
    arrivals = diff_period.rename('period').to_frame().assign(count=1).groupby(
        ['lstg', 'period']).sum().squeeze().astype('int8')
    lstgs = arrivals.index.get_level_values(level='lstg').unique()
    
    # create output dictionary
    y = {}
    for i, lstg in enumerate(periods.index):
        if lstg in lstgs:
            y[i] = arrivals.xs(lstg, level='lstg')

    return y


def get_tf(lstg_start, tf, periods):
    # convert clock into period
    tf = tf.reset_index('clock')
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
    lstgs = tf.index.get_level_values(level='lstg').unique()

    # return output dictionary
    tf_dict = {}
    for i, lstg in enumerate(periods.index):
        if lstg in lstgs:
            tf_dict[i] = tf.xs(lstg, level='lstg')

    return tf_dict


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # function to load file
    load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

    # number of periods
    lstg_start = load_file('lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    periods = get_periods(lstg_start, lstg_end)

    # arrival couns
    thread_start = load_file('clock').xs(1, level='index')
    y = get_y(lstg_start, thread_start, periods)

    # initialize dictionary of input features
    x = load_file('x_lstg')
    x = {k: v.reindex(index=periods.index) for k, v in x.items()}

    # index of first x_clock for each y
    seconds = lstg_start.reindex(index=periods.index)

    # time features
    tf = get_tf(lstg_start, load_file('tf_arrival'), periods)

    return {'y': y, 'periods': periods, 'x': x, 
            'seconds': seconds, 'tf': tf}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'arrival')
    