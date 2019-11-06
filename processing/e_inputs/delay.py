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
    x_time['minutes'] = (clock.dt.hour * 60 + clock.dt.minute) / (24 * 60)
    return x_time.drop('clock', axis=1)


def parse_time_feats_delay(role, idx, z_start, z_role):
    # initialize output
    x_time = pd.DataFrame(index=idx).join(z_start)
    # time of each pseudo-observation
    x_time['clock'] += INTERVAL[role] * x_time.period
    # features from clock
    x_time = add_clock_feats(x_time)
    # differenced time-varying features
    tfdiff = z_role.groupby(['lstg', 'thread', 'index']).diff().dropna()
    tfdiff = tfdiff.rename(lambda x: x + '_diff', axis=1)
    x_time = x_time.join(tfdiff.reindex(idx, fill_value=0))
    return x_time


def parse_fixed_feats_delay(idx, x_lstg, x_thread, x_offer):
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
def process_inputs(part, role):
    # path name function
    getPath = lambda names: '%s/partitions/%s/%s.gz' % \
        (PREFIX, part, '_'.join(names))

    # outcome
    y = load(getPath(['y', 'delay', role]))

    # sort by number of turns
    turns = get_sorted_turns(y)
    y = y.reindex(index=turns.index)

    # fixed features
    x_lstg = load(getPath(['x', 'lstg']))
    x_thread = load(getPath(['x', 'thread']))
    x_offer = load(getPath(['x', 'offer']))
    tf_raw = load(getPath(['tf', 'delay', 'raw', role]))
    x_fixed = parse_fixed_feats_delay(
        y.index, x_lstg, x_thread, x_offer, tf_raw)

    # time features
    z_role = load(getPath(['z', role]))
    x_time = parse_time_feats_delay(role, y.index, z_start, z_role)

    

    return {'y': y.astype('int8', copy=False),
            'turns': turns.astype('uint16', copy=False),
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

    # input dataframes, output processed dataframes
    d = process_inputs(part, role)

    # save featnames and sizes
    if part == 'train_models':
        pickle.dump(get_featnames(d), 
            open('%s/inputs/featnames/delay_%s.pkl' % (PREFIX, role), 'wb'))
        pickle.dump(get_sizes(d), 
            open('%s/inputs/sizes/delay_%s.pkl' % (PREFIX, role), 'wb'))

    # save dictionary of numpy arrays
    dump(convert_to_numpy(d), 
        '%s/inputs/%s/delay_%s.gz' % (PREFIX, part, role))
