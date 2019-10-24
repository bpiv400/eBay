import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def extract_hour_feats(clock):
    df = pd.DataFrame(index=clock.index)
    df['days'] = pd.to_timedelta(clock - pd.to_datetime(START)).dt.days
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    df['hour'] = clock.dt.hour
    return df


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
    # path name function
    getPath = lambda names: '%s/partitions/%s/%s.gz' % \
        (PREFIX, part, '_'.join(names))

    # outcome
    y = load(getPath(['y', 'arrival']))

    # fixed features
    x_fixed = load(getPath(['x', 'lstg']))
    assert(x_fixed.index.equals(y.index))

    # clock features
    N = pd.to_timedelta(pd.to_datetime(END) - pd.to_datetime(START))
    N = int((N.total_seconds()+1) / 3600)
    clock = pd.to_datetime(range(N+30*24), unit='h', origin=START)
    clock = pd.Series(clock, name='clock')
    x_hour = extract_hour_feats(clock).join(clock).set_index('clock')

    # index of first x_hour for each y
    idx_hour = x_fixed.start_date * 24

    return {'y': y.astype('int8', copy=False),
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_hour': x_hour.astype('uint16', copy=False),
            'idx_hour': idx_hour.astype('uint16', copy=False)}


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
