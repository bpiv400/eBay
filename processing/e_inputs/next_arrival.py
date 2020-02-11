from compress_pickle import load
import pandas as pd
from processing.processing_utils import input_partition, load_file, \
    collect_date_clock_feats, get_arrival_times, get_months_since_lstg, \
    init_x, save_files
from processing.processing_consts import CLEAN_DIR, INTERVAL, INTERVAL_COUNTS
from constants import ARRIVAL_PREFIX, MONTH
from featnames import THREAD_COUNT, MONTHS_SINCE_LAST, MONTHS_SINCE_LSTG


def get_interarrival_period(clock):
    # calculate interarrival times in seconds
    df = clock.unstack()
    diff = pd.DataFrame(0.0, index=df.index, columns=df.columns[2:])
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
    y = diff[diff > 0]

    # reindex censored and diff
    censored = censored.reindex(index=y.index)
    diff = diff.reindex(index=y.index)

    # convert y to periods
    y //= INTERVAL[ARRIVAL_PREFIX]

    # replace censored interarrival times negative count of censored buckets
    y.loc[censored] -= INTERVAL_COUNTS[ARRIVAL_PREFIX]

    return y, diff


def get_x_thread_arrival(clock, idx, lstg_start, diff):
    # seconds since START at beginning of arrival window
    seconds = clock.groupby('lstg').shift().dropna().astype(
        'int64').reindex(index=idx)

    # clock features
    clock_feats = collect_date_clock_feats(seconds)

    # thread count so far
    thread_count = pd.Series(seconds.index.get_level_values(level='thread') - 1,
                             index=seconds.index, name=THREAD_COUNT)

    # months since lstg start
    months_since_lstg = get_months_since_lstg(lstg_start, seconds)
    assert (months_since_lstg.max() < 1) & (months_since_lstg.min() >= 0)

    # months since last arrival
    months_since_last = diff.groupby('lstg').shift().fillna(0) / MONTH

    # concatenate into dataframe
    x_thread = pd.concat(
        [clock_feats,
         months_since_lstg.rename(MONTHS_SINCE_LSTG),
         months_since_last.rename(MONTHS_SINCE_LAST),
         thread_count], axis=1)

    return x_thread.astype('float32')


def process_inputs(part):
    # timestamps
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    thread_start = load_file(part, 'clock').xs(1, level='index')

    # arrival times
    clock = get_arrival_times(lstg_start, thread_start, lstg_end)

    # interarrival times
    y, diff = get_interarrival_period(clock)
    idx = y.index

    # listing features
    x = init_x(part, idx, drop_slr=True)

    # add thread features to x['lstg']
    x_thread = get_x_thread_arrival(clock, idx, lstg_start, diff)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    return {'y': y, 'x': x}


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
