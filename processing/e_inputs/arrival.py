import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
    # path name function
    getPath = lambda names: '%s/partitions/%s/%s.gz' % \
        (PREFIX, part, '_'.join(names))

    # outcome
    y = load(getPath(['y', 'arrival']))

    # sort by number of turns
    turns = get_sorted_turns(y)
    y = y.reindex(index=turns.index)

    # initialize dictionary of input features
    x = init_x(getPath, turns.index)

    # clock features by minute
    x_clock = create_x_clock()

    # index of first x_clock for each y
    idx_clock = load(getPath(['lookup']))['start_date'].reindex(
        index=turns.index).astype('int64') * 24 * 60

    # time features
    tf = load(getPath(['tf', 'arrival'])).reindex(
        index=turns.index, level='lstg')

    return {'y': y.astype('int8', copy=False),
            'turns': turns.astype('uint16', copy=False),
            'x': {k: v.astype('float32', copy=False) for k, v in x.items()}, 
            'x_clock': x_clock.astype('float32', copy=False),
            'idx_clock': idx_clock.astype('int64', copy=False),
            'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
    # extract model and outcome from int
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int)
    num = parser.parse_args().num-1

    # partition and outcome
    part = PARTITIONS[num]
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
    