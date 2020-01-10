import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, periods_to_string, \
    load_frames, save_files, load_file, get_arrival, get_idx_x, get_first_index, get_tf
from processing.processing_consts import CLEAN_DIR, INTERVAL, INTERVAL_COUNTS


def get_periods(lstg_start, lstg_end):
# intervals in lstg
    periods = (lstg_end - lstg_start) // INTERVAL['arrival']

    # error checking
    assert periods.max() < INTERVAL_COUNTS['arrival']

    # minimum number of periods is 1
    periods += 1

    return periods


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # number of periods
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    periods = get_periods(lstg_start, lstg_end)
    idx = periods.index

    # arrival counts
    thread_start = load_file(part, 'clock').xs(1, level='index')
    arrival = get_arrival(lstg_start, thread_start)

    # periods in which arrivals are observed
    arrival_periods = periods_to_string(arrival.index, idx)

    # first index of arrivals for each lstg
    idx_arrival = get_first_index(arrival.index, idx)

    # error checking
    assert all(arrival_periods[idx_arrival == -1] == ''.encode('ascii'))
    assert all(idx_arrival[arrival_periods == ''.encode('ascii')] == -1)

    # index of listing features
    idx_x = get_idx_x(part, idx)

    # time features
    tf = load_file(part, 'tf_arrival')
    tf_arrival = get_tf(tf, lstg_start, periods, 'arrival')

    # periods in which non-zero time features are observed
    tf_periods = periods_to_string(tf_arrival.index, idx)

    # first index of time features for each lstg
    idx_tf = get_first_index(tf_arrival.index, idx)

    # error checking
    assert all(tf_periods[idx_tf == -1] == ''.encode('ascii'))
    assert all(idx_tf[tf_periods == ''.encode('ascii')] == -1)

    return {'periods': periods, 
            'arrival': arrival, 
            'arrival_periods': arrival_periods,
            'idx_arrival': idx_arrival,
            'idx_x': idx_x,
            'seconds': lstg_start, 
            'tf': tf_arrival,
            'tf_periods': tf_periods,
            'idx_tf': idx_tf}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'arrival')
    