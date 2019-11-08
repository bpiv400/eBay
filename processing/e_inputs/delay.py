import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def add_past_offers(role, x_fixed, x_offer, tf_raw):
    # last 2 offers
    df = x_offer.join(tf_raw.reindex(
        index=x_offer.index, fill_value=0))
    offer1 = df.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    offer2 = df.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    # drop constant features
    if role == 'byr':
        offer2 = offer2.drop(['auto', 'exp', 'reject'], axis=1)
    elif role == 'slr':
        offer1 = offer1.drop(['auto', 'exp', 'reject'], axis=1)
    # append to x_fixed
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

    # load dataframes
    x_lstg = cat_x_lstg(getPath)
    x_thread = load(getPath(['x', 'thread']))
    x_offer = load(getPath(['x', 'offer']))
    tf_raw = load(getPath(['tf', 'delay', 'raw']))
    clock = load(getPath(['clock']))

    # initialize fixed features
    x_fixed = pd.DataFrame(index=y.index).join(x_lstg).join(x_thread)

    # turn indicators
    x_fixed = add_turn_indicators(x_fixed)

    # add weeks since thread start
    thread_start = clock.xs(1, level='index')
    sec = clock - thread_start.reindex(index=clock.index)
    sec = sec.groupby(['lstg', 'thread']).shift().dropna().astype('int64')
    x_fixed.loc[:, 'days_since_thread'] = sec / (3600 * 24)

    # add last two offers
    x_fixed = add_past_offers(role, x_fixed, x_offer, tf_raw)

    # clock features by minute
    N = pd.to_timedelta(
        pd.to_datetime('2016-12-31 23:59:59') - pd.to_datetime(START))
    N = int((N.total_seconds()+1) / 60)
    clock = pd.to_datetime(range(N), unit='m', origin=START)
    clock = pd.Series(clock, name='clock')
    x_clock = extract_clock_feats(clock).join(clock).set_index('clock')

    # index of first x_clock for each y
    start = x_offer.clock.groupby(['lstg', 'thread']).shift().reindex(
        index=turns.index)
    idx_clock = (start // 60).astype('int64')

    # time features
    tf = load(getPath(['tf', 'delay', 'diff', role]))

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
