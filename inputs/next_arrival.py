import pandas as pd
import numpy as np
from processing.util import collect_date_clock_feats
from inputs.util import get_arrival_times, save_files, get_ind_x
from utils import get_days_since_lstg, input_partition, load_file
from constants import DAY, MAX_DAYS, INTERVAL_ARRIVAL, INTERVAL_CT_ARRIVAL
from featnames import THREAD_COUNT, DAYS_SINCE_LAST, DAYS_SINCE_LSTG, \
    START_TIME, END_TIME, LOOKUP


def get_interarrival_period(arrivals):
    # calculate interarrival times in seconds
    df = arrivals.unstack()
    diff = pd.DataFrame(0.0, index=df.index, columns=df.columns[1:])
    for i in diff.columns:
        diff[i] = df[i] - df[i-1]
    diff = diff.rename_axis(arrivals.index.names[-1], axis=1).stack()

    # original datatype
    diff = diff.astype(arrivals.dtype)

    # indicator for whether observation is last in lstg
    thread = pd.Series(diff.index.get_level_values(level='thread'),
                       index=diff.index)
    last_thread = thread.groupby('lstg').max().reindex(
        index=thread.index, level='lstg')
    censored = thread == last_thread

    # drop interarrivals after BINs
    diff = diff[diff > 0]
    y = diff[diff.index.get_level_values(level='thread') > 1]
    censored = censored.reindex(index=y.index)

    # convert y to periods
    y //= INTERVAL_ARRIVAL

    # replace censored interarrival times negative count of censored buckets
    y.loc[censored] -= INTERVAL_CT_ARRIVAL
    return y, diff


def get_x_thread_arrival(arrivals=None, lstg_start=None, idx=None, diff=None):
    # seconds since START at beginning of arrival window
    seconds = arrivals.groupby('lstg').shift().dropna().astype(
        'int64').reindex(index=idx)

    # clock features
    clock_feats = collect_date_clock_feats(seconds)

    # thread count so far
    thread_num = seconds.index.get_level_values(level='thread')
    thread_count = pd.Series(thread_num - 1,
                             index=seconds.index,
                             name=THREAD_COUNT)

    # days since lstg start
    days_since_lstg = get_days_since_lstg(lstg_start, seconds)
    assert days_since_lstg.max() < MAX_DAYS
    assert days_since_lstg.min() >= 0

    # days since last arrival
    days_since_last = diff.groupby('lstg').shift().dropna() / DAY
    assert np.all(days_since_last.index == idx)

    # concatenate into dataframe
    x_thread = pd.concat(
        [clock_feats,
         days_since_lstg.rename(DAYS_SINCE_LSTG),
         days_since_last.rename(DAYS_SINCE_LAST),
         thread_count], axis=1)

    return x_thread.astype('float32')


def process_inputs(part):
    # data
    clock = load_file(part, 'clock')
    lookup = load_file(part, LOOKUP)
    lstg_start = lookup[START_TIME]
    lstg_end = lookup[END_TIME]

    # arrival times
    arrivals = get_arrival_times(clock=clock,
                                 lstg_start=lstg_start,
                                 lstg_end=lstg_end,
                                 append_last=True)

    # interarrival times
    y, diff = get_interarrival_period(arrivals)

    # thread features
    x = {'thread': get_x_thread_arrival(arrivals=arrivals,
                                        lstg_start=lstg_start,
                                        idx=y.index,
                                        diff=diff)}

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=y.index)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # partition name from command line
    part = input_partition()
    print('%s/next_arrival' % part)

    # create input dictionary
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'next_arrival')


if __name__ == '__main__':
    main()
