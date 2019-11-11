import sys, pickle, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def add_past_offers(role, x_fixed, x_offer, tf_raw):
    # combine offer dataframe and time feats
    df = x_offer.join(tf_raw.reindex(
        index=x_offer.index, fill_value=0))
    # last 2 offers
    offer1 = df.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=x_fixed.index)
    offer2 = df.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=x_fixed.index)
    # drop clock feats from offer1
    toDrop = ['holiday', 'minute_of_day'] + \
                [x for x in offer1.columns if 'dow' in x]
    offer1 = offer1.drop(toDrop , axis=1)
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
    turns = turns[turns > 0]
    y = y.reindex(index=turns.index)

    # load dataframes
    x_lstg = cat_x_lstg(getPath)
    x_thread = load(getPath(['x', 'thread']))
    x_offer = load(getPath(['x', 'offer']))
    tf_raw = load(getPath(['tf', 'role', 'raw']))
    clock = load(getPath(['clock']))
    lstg_start = load(getPath(['lookup'])).start_date.astype(
        'int64') * 24 * 3600

    # initialize fixed features
    x_fixed = pd.DataFrame(index=y.index).join(x_lstg).join(
        x_thread[['byr_hist', 'months_since_lstg']])

    # turn indicators
    x_fixed = add_turn_indicators(x_fixed)

    # add days since thread start
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
    minute = pd.to_datetime(range(N), unit='m', origin=START)
    minute = pd.Series(minute, name='clock')
    x_clock = extract_clock_feats(minute).join(minute).set_index('clock')

    # index of first x_clock for each y
    delay_start = clock.groupby(['lstg', 'thread']).shift().reindex(
        index=turns.index).astype('int64')
    idx_clock = delay_start // 60

    # normalized periods remaining at start of delay period
    remaining = MAX_DAYS * 24 * 3600 - (delay_start - lstg_start)
    remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
        INTERVAL['slr']
    remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
        INTERVAL['byr']
    remaining /= len(y.columns) # normalize by total number of intervals
    remaining = np.minimum(remaining, 1)

    # time features
    tf = load(getPath(['tf', 'delay', 'diff', role]))

    return {'y': y.astype('int8', copy=False),
            'turns': turns.astype('uint16', copy=False),
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_clock': x_clock.astype('uint16', copy=False),
            'idx_clock': idx_clock.astype('int64', copy=False),
            'remaining': remaining.astype('float32', copy=False),
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
