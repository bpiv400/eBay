from compress_pickle import load
import pandas as pd
import numpy as np
from processing.processing_utils import input_partition, load_file, collect_date_clock_feats
from processing.e_inputs.inputs_utils import get_interarrival_period, init_x, save_files
from utils import get_months_since_lstg
from processing.processing_consts import CLEAN_DIR
from constants import MONTH
from featnames import THREAD_COUNT, MONTHS_SINCE_LAST, MONTHS_SINCE_LSTG


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
    months_since_last = diff.groupby('lstg').shift().dropna() / MONTH
    assert np.all(months_since_last.index == idx)

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
    x = init_x(part, idx)

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
