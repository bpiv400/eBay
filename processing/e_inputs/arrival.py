import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
    # path name function
    getPath = lambda names: PARTS_DIR + '%s/%s.gz' % \
        (part, '_'.join(names))

    # outcome
    y = load(getPath(['y', 'arrival']))
    idx = y.index

    # initialize dictionary of input features
    x = init_x(part, idx)

    # clock features by minute
    x_clock = create_x_clock()

    # index of first x_clock for each y
    idx_clock = load(getPath(['lookup']))['start_date'].reindex(
        index=idx).astype('int64') * 24 * 60

    # time features
    tf = load(getPath(['tf', 'arrival'])).reindex(
        index=idx, level='lstg')

    return {'y': y.astype('int8', copy=False),
            'x': {k: v.astype('float32', copy=False) for k, v in x.items()}, 
            'x_clock': x_clock.astype('float32', copy=False),
            'idx_clock': idx_clock.astype('int64', copy=False),
            'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
    # extract model and outcome from int
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    part = parser.parse_args().part
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
    