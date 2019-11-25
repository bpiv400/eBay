import sys, pickle, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *
from processing.e_inputs.inputs import Inputs


def get_x_fixed(idx, x_lstg, x_thread, x_offer, role):
    # reindex to listings in y
    x_fixed = x_lstg.reindex(index=idx, level='lstg')
    # join with thread features
    x_fixed = x_fixed.join(x_thread.months_since_lstg)
    x_fixed = x_fixed.join(x_thread.byr_hist.astype('float32') / 10)
    # merged offer and time feats, excluding index 0
    threads = idx.droplevel(level='index').unique()
    df = pd.DataFrame(index=threads).join(x_offer)
    # add in offers, excluding final turn
    for i in range(1, max(IDX[role])):
        print(i)
        # get features at index i
        offer = df.xs(i, level='index').reindex(index=idx)
        # if turn 1, drop days and delay
        if i == 1:
            offer = offer.drop(['days', 'delay'], axis=1)
        else:
            # set features to 0 if turn i in future
            future = i >= offer.index.get_level_values(level='index')
            offer.loc[future, df.dtypes == 'bool'] = False
            offer.loc[future, df.dtypes != 'bool'] = 0
        # if buyer turn, drop auto, exp, reject
        if i in IDX['byr']:
            offer = offer.drop(['auto', 'exp', 'reject'], axis=1)
        # add turn number to feat names and add to x_fixed
        x_fixed = x_fixed.join(
            offer.rename(lambda x: x + '_%d' % i, axis=1))
    # add turn indicators and return
    return add_turn_indicators(x_fixed)


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
    clock = load(getPath(['clock']))
    lstg_start = load(getPath(['lookup'])).start_date.astype(
        'int64') * 24 * 3600

    # initialize fixed features
    x_fixed = get_x_fixed(y.index, x_lstg, x_thread, x_offer, role)

    # clock features by minute
    x_clock = create_x_clock()

    # index of first x_clock for each y
    delay_start = clock.groupby(['lstg', 'thread']).shift().reindex(
        index=turns.index).astype('int64')
    idx_clock = delay_start // 60

    # normalized periods remaining at start of delay period
    remaining = MAX_DAYS * 24 * 3600 - (delay_start - lstg_start)
    remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
        MAX_DELAY['slr']
    remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
        MAX_DELAY['byr']
    remaining = np.minimum(remaining, 1)

    # time features
    tf = load(getPath(['tf', 'delay', 'diff', role]))

    return {'y': y.astype('int8', copy=False),
            'turns': turns.astype('uint16', copy=False),
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_clock': x_clock.astype('float32', copy=False),
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
        pickle.dump(get_sizes(d, model), 
            open('%s/inputs/sizes/delay_%s.pkl' % (PREFIX, role), 'wb'))

    # create dictionary of numpy arrays
    d = convert_to_numpy(d)

    # save as dataset
    dump(Inputs(d, model), '%s/inputs/%s/delay_%s.gz' % (PREFIX, part, role))

    # save small dataset
    if part == 'train_models':
        small = create_small(d)
        dump(Inputs(small, model), 
            '%s/inputs/small/delay_%s.gz' % (PREFIX, role))
