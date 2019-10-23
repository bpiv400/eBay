import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def extract_hour_feats(clock):
    df = pd.DataFrame(index=clock.index)
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
    y = load(getPath(['y', 'arrival'])).unstack(fill_value=-1)

    # fixed features
    x_fixed = load(getPath(['x_fixed']))
    assert(x_fixed.index.equals(y.index))

    # index of x_fixed for each y
    lookup = np.array(range(len(x_fixed.index)))
    counts = y.groupby('lstg').count().values
    idx_fixed = np.repeat(lookup, counts)

    # clock features
    N = pd.to_timedelta(pd.to_datetime(END) - pd.to_datetime(START)).hours
    clock = pd.to_datetime(range(N+30*24+1), unit='h', origin=START)
    x_hour = pd.Series(clock, name='clock')
    x_hour = extract_hour_feats(x_hour).join(x_hour).set_index('clock')

    # index of x_hour for each y
    period = y.reset_index('period')['period']
    idx_hour = (period + x_fixed.start_date * 24).values

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

    # out path
    path = lambda x: '%s/%s/%s/arrival.gz' % (PREFIX, x, part)

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save featnames and sizes
    if part == 'train_models':
        pickle.dump(get_featnames(d),
            open('%s/featnames/con_%s.pkl' % (PREFIX, role), 'wb'))
        pickle.dump(get_sizes(d),
            open('%s/sizes/con_%s.pkl' % (PREFIX, role), 'wb'))

    # save dictionary of numpy arrays
    dump(convert_to_numpy(d), 
        '%s/inputs/%s/arrival.gz' % (PREFIX, part))
