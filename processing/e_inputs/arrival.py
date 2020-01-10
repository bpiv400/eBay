import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, reshape_indices, \
    load_frames, save_files, load_file, get_arrival, get_idx_x, get_tf
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

    # periods and arrival indices with index idx
    arrival_periods, idx_arrival = reshape_indices(arrival.index, idx)

    # index of listing features
    idx_x = get_idx_x(part, idx)

    # time features
    tf = load_file(part, 'tf_arrival')
    tf_arrival = get_tf(tf, lstg_start, periods, 'arrival')

    # periods and tf indices with index idx
    tf_periods, idx_tf = reshape_indices(tf_arrival.index, idx)

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
    