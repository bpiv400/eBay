import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


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


def parse_fixed_feats_delay(model, idx, x_lstg, x_thread, x_offer):
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
    return x_fixed


# loads data and calls helper functions to construct training inputs
def process_inputs(part, model):
    # path name function
    getPath = lambda names: '%s/partitions/%s/%s.gz' % \
        (PREFIX, part, '_'.join(names))

    # outcome
    y = load(getPath(['y', model, 'delay']))

    # time features
    z_start = load(getPath(['z', 'start']))
    z_role = load(getPath(['z', model]))
    x_time = parse_time_feats_delay(model, y.index, z_start, z_role)

    # unstack y
    y = y.astype('float32').unstack()
    y[y.isna()] = -1

    # fixed features
    x_lstg = cat_x_lstg(part)
    x_thread = load(getPath(['x', 'thread']))
    x_offer = load(getPath(['x', 'offer']))
    x_fixed = parse_fixed_feats_delay(
    	model, y.index, x_lstg, x_thread, x_offer)

    return {'y': y.astype('int8', copy=False), 
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_time': x_time.astype('float32', copy=False)}


if __name__ == '__main__':
    # extract model and outcome from int
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int)
    num = parser.parse_args().num-1

    # partition and outcome
    part = PARTITIONS[num // 2]
    role = 'slr' if num % 2 else 'byr'
    model = 'delay_%s' % role
    print('%s/%s' % (part, model))

    # out path
    path = lambda x: '%s/%s/%s/delay_%s.gz' % (PREFIX, x, part, role)

    # input dataframes, output processed dataframes
    d = process_inputs(part, model)

    # save featnames and sizes
    if part == 'train_models':
        pickle.dump(get_featnames(d), 
            open('%s/featnames/con_%s.pkl' % (PREFIX, role), 'wb'))
        pickle.dump(get_sizes(d), 
            open('%s/sizes/con_%s.pkl' % (PREFIX, role), 'wb'))

    # save dictionary of numpy arrays
    dump(convert_to_numpy(d), 
        '%s/inputs/%s/delay_%s.gz' % (PREFIX, part, role))
