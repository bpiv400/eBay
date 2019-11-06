import sys, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_arrival_time_feats(lookup, tf):
    # add period to tf_arrival
    tf = tf.reset_index('clock')
    tf = tf.join((lookup.start_date * 24 * 3600).rename('start_time'))
    tf['period'] = (tf.clock - tf.start_time) // 3600
    tf = tf.drop(['clock', 'start_time'], axis=1)
    # increment period by 1; time feats are up to t-1
    tf['period'] += 1
    # drop periods beyond censoring threshold
    tf = tf[tf.period < MAX_DAYS * 24]
    # sum count features by period and return
    return tf.groupby(['lstg', 'period']).sum()


def get_y_arrival(lookup, threads):
    # time_stamps
    t0 = lookup.start_date * 24 * 3600
    diff = pd.to_timedelta(threads.start_time - t0, unit='s')
    end = pd.to_timedelta(lookup.end_time - t0, unit='s')
    # convert to hours
    diff = (diff.dt.total_seconds() // 3600).astype('uint16')
    end = (end.dt.total_seconds() // 3600).astype('uint16')
    # censor to first 31 days
    diff = diff.loc[diff < MAX_DAYS * 24]
    end.loc[end >= MAX_DAYS * 24] = MAX_DAYS * 24 - 1
    # count of arrivals by hour
    hours = diff.rename('period').to_frame().assign(count=1)
    arrivals = hours.groupby(['lstg', 'period']).sum()
    arrivals = arrivals.squeeze().astype('int8')
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
    return df


def get_x_thread(threads):
    x_thread = threads['byr_hist']
    # decrement 100th percentile by epsilon
    x_thread.loc[x_thread == 1] -= 1e-16
    # convert to quantiles and return
    return np.floor(HIST_QUANTILES * x_thread) / HIST_QUANTILES


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load data
    lookup = load(PARTS_DIR + '%s/lookup.gz' % part)
    threads = load(CLEAN_DIR + 'threads.gz').reindex(
        index=idx, level='lstg')
    tf = load_frames('tf_lstg_arrival').reindex(
        index=idx, level='lstg')

    # thread variables
    print('x_thread')
    x_thread = get_x_thread(threads)
    dump(x_thread, path('x_thread'))

    # time feats
    print('tf_arrival')
    tf_arrival = get_arrival_time_feats(lookup, tf)
    dump(tf_arrival, path('tf_arrival'))

    # # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lookup, threads)
    dump(y_arrival, path('y_arrival'))