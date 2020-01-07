import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, \
    load_frames, save_files, load_file, get_tf
from processing.processing_consts import CLEAN_DIR, INTERVAL, INTERVAL_COUNTS


def get_periods(lstg_start, lstg_end):
# intervals in lstg
    periods = (lstg_end - lstg_start) // INTERVAL['arrival']

    # error checking
    assert periods.max() < INTERVAL_COUNTS['arrival']

    # minimum number of periods is 1
    periods += 1

    return periods


def get_y(lstg_start, thread_start):
    # intervals until thread
    thread_periods = (thread_start - lstg_start) // INTERVAL['arrival']

    # error checking
    assert thread_periods.max() < INTERVAL_COUNTS['arrival']

    # count of arrivals by interval
    y = thread_periods.rename('period').to_frame().assign(
        count=1).groupby(['lstg', 'period']).sum().squeeze()

    return y


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # number of periods
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    periods = get_periods(lstg_start, lstg_end)

    # listing features
    x = load_file(part, 'x_lstg')

    # arrival counts
    thread_start = load_file(part, 'clock').xs(1, level='index')
    y = get_y(lstg_start, thread_start)

    # time features
    tf = load_file(part, 'tf_arrival')
    tf_arrival = get_tf(tf, lstg_start, periods, 'arrival')

    return {'periods': periods, 'y': y, 'x': x,
            'seconds': lstg_start, 'tf': tf_arrival}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'arrival')
    