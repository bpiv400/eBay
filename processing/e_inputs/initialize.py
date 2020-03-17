import argparse
import numpy as np
import pandas as pd
from processing.processing_utils import load_file, get_x_thread, \
    init_x, save_files, get_y_con, check_zero, calculate_remaining
from constants import IDX, BYR_PREFIX, SLR_PREFIX
from featnames import DELAY, AUTO, EXP, REJECT, CON, NORM, SPLIT, MSG, INT_REMAINING, TIME_FEATS


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
    if i > 1:
        offer.loc[i > turn, :] = 0.0
    # for current turn, set feats to 0
    curr = i == turn
    offer.loc[curr, [CON, NORM, SPLIT, AUTO, EXP, REJECT, MSG]] = 0.0
    return offer


def get_x_offer(offers, idx, role):
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

        # set unseen feats to 0 and add turn indicators
        offer = set_zero_feats(offer, i)
        offer = add_turn_indicators(offer)

        # set censored time feats to zero
        if i > 1 and role == SLR_PREFIX:
            censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
            offer.loc[censored, TIME_FEATS] = 0.0

        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')

    return x_offer


# loads data and calls helper functions to construct train inputs
def process_inputs(part, role):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')

    # outcome and master index
    df = offers[offers.index.isin(IDX[role], level='index')]
    y = get_y_con(df)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # remove auto accept/reject features from x['lstg'] for buyer models
    if role == BYR_PREFIX:
        x['lstg'].drop(['auto_decline', 'auto_accept', 'has_decline', 'has_accept'],
                       axis=1, inplace=True)

    # thread features
    x_thread = get_x_thread(threads, idx)
    x_thread = add_turn_indicators(x_thread)
    x_thread[INT_REMAINING] = calculate_remaining(part, idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx, role))

    # error checking
    check_zero(x)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, role, delay = args.part, args.role, args.delay
    assert role in [BYR_PREFIX, SLR_PREFIX]
    if role == 'byr' or delay:
        raise NotImplementedError()
    else:
        name = 'init_{}'.format(role)
    print('%s/%s' % (part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, role)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
