import sys
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import load_frames, get_partition, sort_by_turns


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


def get_y_arrival(lstg_start, lstg_end, thread_start):
    # time_stamps
    diff = pd.to_timedelta(thread_start - lstg_start, unit='s')
    end = pd.to_timedelta(lstg_end - lstg_start, unit='s')
    # convert to intervals
    diff = (diff.dt.total_seconds() // INTERVAL['arrival']).astype('uint16')
    end = (end.dt.total_seconds() // INTERVAL['arrival']).astype('uint16')
    # censor to first 31 days
    diff = diff.loc[diff < INTERVAL_COUNTS['arrival']]
    end.loc[end >= INTERVAL_COUNTS['arrival']] = INTERVAL_COUNTS['arrival'] - 1
    # count of arrivals by interval
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
    # sort by turns and return
    return sort_by_turns(df)


if __name__ == "__main__":
    # partition and corresponding indices
    part = input_partition()
    idx, path = get_partition(part)

    # load data
    lookup = load(PARTS_DIR + '%s/lookup.gz' % part)
    thread_start = load(CLEAN_DIR + 'offers.pkl').reindex(
        index=idx, level='lstg').clock.xs(1, level='index')
    tf = load_frames('tf_arrival').reindex(
        index=idx, level='lstg')

    # time feats
    print('tf_arrival')
    tf_arrival = get_arrival_time_feats(lookup.start_time, tf)
    dump(tf_arrival, path('tf_arrival'))

    # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lookup.start_time, lookup.end_time, thread_start)
    dump(y_arrival, path('y_arrival'))
