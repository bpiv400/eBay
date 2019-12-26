import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import create_x_clock, convert_to_numpy, \
    get_featnames, get_sizes, create_small


def get_y_arrival(lstg_start, lstg_end, thread_start):
    # time_stamps
    diff = pd.to_timedelta(thread_start - lstg_start, unit='s')
    end = pd.to_timedelta(lstg_end - lstg_start, unit='s')
    # convert to intervals
    diff = (diff.dt.total_seconds() // INTERVAL['arrival']).astype('int64')
    end = (end.dt.total_seconds() // INTERVAL['arrival']).astype('int64')
    # count of arrivals by interval
    arrivals = diff.rename('period').to_frame().assign(count=1).groupby(
        ['lstg', 'period']).sum().squeeze().astype('int8')
    # error checking
    assert diff.max() < INTERVAL_COUNTS['arrival']
    assert end.max() < INTERVAL_COUNTS['arrival']
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


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # function to load file
    load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

    # outcome
    lstg_start = load_file('lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    thread_start = load_file('clock').xs(1, level='index')
    y = get_y_arrival(lstg_start, lstg_end, thread_start)
    idx = y.index

    # initialize dictionary of input features
    x = load_file('x_lstg')
    x = {k: v.reindex(index=idx) for k, v in x.items()}

    # clock features by minute
    x_clock = create_x_clock()

    # index of first x_clock for each y
    idx_clock = load_file('lookup').start_time.reindex(index=idx)

    # time features
    tf = load_file('tf_arrival').reindex(index=idx, level='lstg')

    return {'y': y.astype('int8', copy=False),
            'x': {k: v.astype('float32', copy=False) for k, v in x.items()}, 
            'x_clock': x_clock.astype('float32', copy=False),
            'idx_clock': idx_clock.astype('int64', copy=False),
            'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save featnames and sizes
    if part == 'train_models':
        pickle.dump(get_featnames(d),
            open('%s/inputs/featnames/arrival.pkl' % PREFIX, 'wb'))
        pickle.dump(get_sizes(d, 'arrival'),
            open('%s/inputs/sizes/arrival.pkl' % PREFIX, 'wb'))

    # create dictionary of numpy arrays
    d = convert_to_numpy(d)

    # save as dataset
    dump(d, '%s/inputs/%s/arrival.gz' % (PREFIX, part))

    # save small dataset
    if part == 'train_models':
        dump(create_small(d), '%s/inputs/small/arrival.gz' % PREFIX)
    