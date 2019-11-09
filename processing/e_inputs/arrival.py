import sys, pickle, os
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

    # fixed features
    x_fixed = cat_x_lstg(getPath).reindex(index=turns.index)

    # clock features
    N = pd.to_timedelta(pd.to_datetime(END) - pd.to_datetime(START))
    N = int((N.total_seconds()+1) / 3600)
    clock = pd.to_datetime(range(N+30*24), unit='h', origin=START)
    clock = pd.Series(clock, name='clock')
    x_clock = extract_clock_feats(clock).join(clock).set_index('clock')

    # index of first x_clock for each y
    start = load(getPath(['lookup']))['start_date'].reindex(
        index=turns.index)
    idx_clock = (start * 24).astype('int64')

    # time features
    tf = load(getPath(['tf', 'arrival'])).reindex(
        index=turns.index, level='lstg')

    return {'y': y.astype('int8', copy=False),
            'turns': turns.astype('uint16', copy=False),
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_clock': x_clock.astype('uint16', copy=False),
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
        pickle.dump(get_sizes(d),
            open('%s/inputs/sizes/arrival.pkl' % PREFIX, 'wb'))

    # save dictionary of numpy arrays
    dump(convert_to_numpy(d), 
        '%s/inputs/%s/arrival.gz' % (PREFIX, part))
