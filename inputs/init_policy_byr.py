import argparse
import numpy as np
import pandas as pd
from inputs.inputs_utils import save_files, check_zero, get_x_thread
from utils import get_remaining, load_file, init_x
from constants import IDX, MAX_DELAY, BYR_PREFIX, SLR_PREFIX, \
    CON_MULTIPLIER, TRAIN_MODELS, VALIDATION, TEST
from featnames import DELAY, AUTO, EXP, CON, NORM, SPLIT, REJECT, \
    MSG, INT_REMAINING, TIME_FEATS, OUTCOME_FEATS, CLOCK_FEATS, \
    START_TIME


def get_y(df):
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # add in waiting before make first offer

    return y


def calculate_remaining(lstg_start=None, clock=None, idx=None):
    # start of delay period
    delay_start = clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')

    # remaining is turn-specific
    remaining = pd.Series(np.nan, index=idx)
    for turn in idx.unique(level='index'):
        turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
        mask = idx.get_level_values(level='index') == turn
        remaining.loc[mask] = get_remaining(
            lstg_start, turn_start, MAX_DELAY[turn])

    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)

    return remaining


def add_turn_indicators(df):
    """
    Appends turn indicator variables to offer matrix
    :param df: dataframe with index ['lstg', 'thread', 'index'].
    :return: dataframe with turn indicators appended
    """
    indices = np.sort(np.unique(df.index.get_level_values('index')))
    for i in range(len(indices) - 1):
        ind = indices[i]
        df['t{}'.format(ind)] = df.index.isin([ind], level='index')
    return df


# sets unseen feats to 0
def set_zero_feats(offer, i):
    # turn number
    turn = offer.index.get_level_values(level='index')

    # all features are zero for future turns
    offer.loc[i > turn, :] = 0.

    # for current turn, set non-delay outcome feats to 0
    curr = i == turn
    assert (offer.loc[curr, AUTO] == 0).all()
    offer.loc[curr, [EXP, CON, NORM, SPLIT, REJECT, MSG]] = 0.

    # when choosing delay, set other current-turn feats to 0
    if delay:
        # for seller, set feats to 0 in current turn
        if role == SLR_PREFIX:
            offer.loc[i == turn, :] = 0.
        # for buyer, set feats to 0, except clock feats in turn 1
        else:
            offer.loc[i == turn, OUTCOME_FEATS] = 0.
            if i > 1:
                offer.loc[i == turn, CLOCK_FEATS] = 0.

    return offer


def get_x_offer(offers, idx, role, delay):
    # initialize dictionary of offer features
    x_offer = {}

    # dataframe of offer features for relevant threads
    threads = idx.droplevel(level='index').unique()
    offers = pd.DataFrame(index=threads).join(offers)

    # drop time feats from buyer models
    if role == BYR_PREFIX:
        offers.drop(TIME_FEATS, axis=1, inplace=True)

    # turn features
    for i in range(1, max(IDX[role]) + 1):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0).astype('float32')

        # set unseen feats to 0
        offer = set_zero_feats(offer, i, role, delay)

        # set censored time feats to zero
        if i > 1 and role == SLR_PREFIX:
            censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
            offer.loc[censored, TIME_FEATS] = 0.0

        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')

    # error checking
    check_zero(x_offer)

    return x_offer


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    clock = load_file(part, 'clock')

    # restrict by role, drop auto replies
    df = clock[clock.index.isin(IDX[BYR_PREFIX], level='index')]
    df = df.to_frame().join(offers)

    # outcome and master index
    y = get_y(df)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # remove auto accept/reject features from x['lstg'] for buyer models
    x['lstg'].drop(['auto_decline', 'auto_accept',
                    'has_decline', 'has_accept'],
                   axis=1, inplace=True)

    # thread features
    x_thread = get_x_thread(threads, idx)
    x_thread = add_turn_indicators(x_thread)

    # remaining
    lstg_start = load_file(part, 'lookup')[START_TIME]
    clock = load_file(part, 'clock')
    x_thread[INT_REMAINING] = \
        calculate_remaining(lstg_start=lstg_start,
                            clock=clock,
                            idx=idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx))

    return {'y': y, 'x': x}


def input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    part = parser.parse_args().part
    part, delay = args.part, args.delay
    return part, delay


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    part = parser.parse_args().part
    name = 'init_policy_{}_delay'.format(BYR_PREFIX)
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
