import pandas as pd
import numpy as np
from processing.processing_utils import input_partition, init_x, \
    collect_date_clock_feats, get_obs_outcomes
from processing.e_inputs.inputs_utils import get_arrival_times, save_files
from utils import get_months_since_lstg
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import MONTH
from featnames import THREAD_COUNT, MONTHS_SINCE_LAST, MONTHS_SINCE_LSTG


def get_interarrival_period(d):
    # arrival times
    clock = get_arrival_times(d, append_last=True)

    # calculate interarrival times in seconds
    df = clock.unstack()
    diff = pd.DataFrame(0.0, index=df.index, columns=df.columns[1:])
    for i in diff.columns:
        diff[i] = df[i] - df[i - 1]

    # restack
    diff = diff.rename_axis(clock.index.names[-1], axis=1).stack()

    # original datatype
    diff = diff.astype(clock.dtype)

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
    y //= INTERVAL[1]

    # replace censored interarrival times negative count of censored buckets
    y.loc[censored] -= INTERVAL_COUNTS[1]

    return y, diff


def get_x_thread_arrival(d, idx, diff):
    # seconds since START at beginning of arrival window
    seconds = d['clock'].groupby('lstg').shift().dropna().astype(
        'int64').reindex(index=idx)

    # clock features
    clock_feats = collect_date_clock_feats(seconds)

    # thread count so far
    thread_count = pd.Series(seconds.index.get_level_values(level='thread') - 1,
                             index=seconds.index, name=THREAD_COUNT)

    # months since lstg start
    months_since_lstg = get_months_since_lstg(d['lstg_start'], seconds)
    assert (months_since_lstg.max() < 1) & (months_since_lstg.min() >= 0)

    # months since last arrival
    months_since_last = diff.groupby('lstg').shift().dropna() / MONTH
    assert np.all(months_since_last.index == idx)

    # concatenate into dataframe
    x_thread = pd.concat(
        [clock_feats,
         months_since_lstg.rename(MONTHS_SINCE_LSTG),
         months_since_last.rename(MONTHS_SINCE_LAST),
         thread_count], axis=1)

    return x_thread.astype('float32')


def process_inputs(d, part):
    # interarrival times
    y, diff = get_interarrival_period(d)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # add thread features to x['lstg']
    x_thread = get_x_thread_arrival(d, idx, diff)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    return {'y': y, 'x': x}


def main():
    # partition name from command line
    part = input_partition()
    print('%s/next_arrival' % part)

    # dictionary of components
    obs = get_obs_outcomes(part, timestamps=True)

    # create input dictionary
    d = process_inputs(obs, part)

    # save various output files
    save_files(d, part, 'next_arrival')


if __name__ == '__main__':
    main()
