import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import create_x_clock, convert_to_numpy, \
    get_featnames, get_sizes, create_small


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # function to load file
    load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

    # outcome
    y = load_file('y_arrival')
    idx = y.index

    # initialize dictionary of input features
    x = load_file('x_lstg').reindex(index=idx)

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
    