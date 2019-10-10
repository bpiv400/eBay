import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *


def add_clock_feats(x_time):
    clock = pd.to_datetime(x_time.clock, unit='s', origin=START)
    # US holiday indicator
    x_time['holiday'] = clock.isin(HOLIDAYS)
    # day of week indicator
    for i in range(6):
        x_time['dow' + str(i)] = clock.dt.dayofweek == i
    # minute in day
    x_time['minutes'] = clock.dt.hour * 60 + clock.dt.minute
    return x_time.drop('clock', axis=1)


def parse_time_feats_delay(model, idx, z_start, z_role):
    # initialize output
    x_time = pd.DataFrame(index=idx).join(z_start)
    # add period
    x_time['period'] = idx.get_level_values('period')
    # time of each pseudo-observation
    x_time['clock'] += INTERVAL[model] * x_time.period
    # features from clock
    x_time = add_clock_feats(x_time)
    # differenced time-varying features
    tfdiff = z_role.groupby(['lstg', 'thread', 'index']).diff().dropna()
    tfdiff = tfdiff.rename(lambda x: x + '_diff', axis=1)
    x_time = x_time.join(tfdiff.reindex(idx, fill_value=0))
    return x_time


def add_turn_indicators(df):
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


def parse_fixed_feats_delay(model, idx, x_lstg, x_thread, x_offer, z_role):
    # lstg and byr attributes
    x_fixed = pd.DataFrame(index=idx).join(x_lstg).join(x_thread)
	# turn indicators
    x_fixed = add_turn_indicators(x_fixed)
    # last 2 offers
    drop = [c for c in x_offer.columns if c.endswith('_diff')]
    df = x_offer.drop(drop, axis=1)
    offer1 = df.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    offer2 = df.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    x_fixed = x_fixed.join(offer1.rename(
        lambda x: x + '_other', axis=1))
    x_fixed = x_fixed.join(offer2.rename(
        lambda x: x + '_last', axis=1))
    # time-varying features
    x_fixed = x_fixed.join(z_role.xs(0, level='period').reindex(
        index=x_fixed.index, fill_value=0.0))
    # drop columns with zero variation
    keep = x_fixed.max() > x_fixed.min()
    x_fixed = x_fixed.loc[:, x_fixed.columns[keep]]
    return x_fixed


# loads data and calls helper functions to construct training inputs
def process_inputs(part, model):
    # path name function
    getPath = lambda names: \
    	'data/partitions/%s/%s.gz' % (part, '_'.join(names))

    # outcome
    y = load(getPath(['y', model, 'delay']))
    idx = y.index
    y = y.astype('float32').unstack()
    y[y.isna()] = -1

    # load in other dataframes
    x_lstg = cat_x_lstg(part)
    x_thread = load(getPath(['x', 'thread']))
    x_offer = load(getPath(['x', 'offer']))
    z_start = load(getPath(['z', 'start']))
    z_role = load(getPath(['z', model]))

    # fixed features
    x_fixed = parse_fixed_feats_delay(
    	model, y.index, x_lstg, x_thread, x_offer, z_role)

    # time features
    x_time = parse_time_feats_delay(
        model, idx, z_start, z_role)

    return y, x_fixed, x_time


def get_sizes(y, x_fixed, x_time):
    sizes = {}
    # number of observations
    sizes['N'] = len(x_fixed.index)
    # fixed inputs
    sizes['fixed'] = len(x_fixed.columns)
    # output parameters
    sizes['out'] = 1
    # RNN parameters
    sizes['steps'] = len(y.columns)
    sizes['time'] = len(x_time.columns)
    return sizes


if __name__ == '__main__':
    # extract model and outcome from int
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int)
    num = parser.parse_args().num-1

    # partition and outcome
    part = PARTITIONS[num // 2]
    model = 'slr' if num % 2 else 'byr'
    outfile = lambda x: 'data/inputs/%s/%s_delay.pkl' % (x, model)
    print('Model: %s' % model)
    print('Outcome: delay')
    print('Partition: %s' % part)

    # input dataframes, output processed dataframes
    y, x_fixed, x_time = process_inputs(part, model)

    # save featnames and sizes once
    if part == 'train_models':
    	# save featnames
    	featnames = {'x_fixed': x_fixed.columns, 'x_time': x_time.columns}
    	pickle.dump(featnames, open(outfile('featnames'), 'wb'))

    	# get data size parameters and save
    	sizes = get_sizes(y, x_fixed, x_time)
    	pickle.dump(sizes, open(outfile('sizes'), 'wb'))

    # convert to numpy arrays, save in hdf5
    path = 'data/inputs/%s/%s_delay.hdf5' % (part, model)
    f = h5py.File(path, 'w')

    # y
    f.create_dataset('y', data=y.to_numpy().astype('int8'), dtype='int8')

    # x_fixed
    f.create_dataset('x_fixed', data=x_fixed.to_numpy().astype('float32'),
        dtype='float32')

    # x_time
    arrays = []
    for c in x_time.columns:
    	array = x_time[c].astype('float32').unstack().reindex(
    		index=y.index).to_numpy()
    	arrays.append(np.expand_dims(array, axis=2))
    arrays = np.concatenate(arrays, axis=2)
    f.create_dataset('x_time', data=arrays, dtype='float32')
    	
    f.close()