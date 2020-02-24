import argparse
import numpy as np
import pandas as pd
from processing.processing_consts import MAX_DELAY, INTERVAL, INTERVAL_COUNTS
from processing.processing_utils import load_file, get_x_thread, \
    init_x, save_files
from constants import IDX, DAY, BYR_PREFIX, SLR_PREFIX, PARTITIONS
from featnames import CON, NORM, SPLIT, MSG, AUTO, EXP, REJECT, DAYS, DELAY, \
    INT_REMAINING, TIME_FEATS
from utils import get_remaining


def get_x_offer(offers, idx, outcome, role, turn):
    # initialize dictionary of offer features
    x_offer = {}
    # dataframe of offer features for relevant threads
    offers = pd.DataFrame(index=idx).join(offers)
    # last turn to include
    if outcome == DELAY:
        last = turn - 1
    elif outcome == CON and turn == 1:
        last = 0
    else:
        last = turn
    # turn features
    for i in range(1, last + 1):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0).astype('float32')
        # set unseen feats to 0
        if i == turn:
            assert outcome != DELAY
            offer[MSG] = 0
            if outcome == CON:
                offer[[CON, NORM, SPLIT, AUTO, EXP, REJECT]] = 0.0
        # drop time feats from buyer models
        if role == BYR_PREFIX:
            offer = offer.drop(TIME_FEATS, axis=1)
        # set censored time feats to zero
        else:
            if i > 1:
                censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
                offer.loc[censored, TIME_FEATS] = 0.0
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')
    return x_offer


def get_y_con(df):
    # drop zero delay and expired offers
    mask = ~df[AUTO] & ~df[EXP]
    # concession is an int from 0 to 100
    return (df.loc[mask, CON] * 100).astype('int8')


def get_y_msg(df, role):
    # for buyers, drop accepts and rejects
    if role == BYR_PREFIX:
        mask = (df[CON] > 0) & (df[CON] < 1)
    # for sellers, drop accepts, expires, and auto responses
    else:
        mask = ~df[EXP] & ~df[AUTO] & (df[CON] < 1)
    return df.loc[mask, MSG]


def get_y_delay(df, role):
    # convert to seconds
    delay = np.round(df[DAYS] * DAY).astype('int64')

    # drop zero delays
    delay = delay[delay > 0]
    df = df.reindex(index=delay.index)

    # error checking
    assert delay.max() <= MAX_DELAY[role]
    if role == BYR_PREFIX:
        assert delay.xs(7, level='index').max() <= MAX_DELAY[SLR_PREFIX]

    # convert to periods
    delay //= INTERVAL[role]

    # replace expired delays with last index
    assert np.all(df.loc[df[DELAY] == 1, EXP])
    delay.loc[delay == INTERVAL_COUNTS[role]] = -1

    # replace censored delays with negative index
    delay.loc[df[EXP] & (df[DELAY] < 1)] -= INTERVAL_COUNTS[role]

    return delay


def calculate_remaining(part, idx, role):
    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time.reindex(
        index=idx, level='lstg')
    delay_start = load_file(part, 'clock').groupby(
        ['lstg', 'thread']).shift().reindex(index=idx).astype('int64')

    # maximal delay
    max_delay = pd.Series(MAX_DELAY[role], index=idx)
    max_delay.loc[max_delay.index.isin([7], level='index')] = \
        MAX_DELAY[BYR_PREFIX + '_7']

    # remaining calculation
    remaining = get_remaining(lstg_start, delay_start, max_delay)

    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)

    return remaining


def check_zero(offer, cols):
    for c in cols:
        assert offer[c].max() == 0
        assert offer[c].min() == 0


# loads data and calls helper functions to construct train inputs
def process_inputs(part, outcome, turn):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    df = offers.xs(turn, level='index')

    # role
    role = BYR_PREFIX if turn in IDX[BYR_PREFIX] else SLR_PREFIX

    # y and master index
    if outcome == CON:
        y = get_y_con(df)
        if turn == 7:
            y = y == 100
    elif outcome == MSG:
        y = get_y_msg(df, role)
    else:
        y = get_y_delay(df, role)
    idx = y.index

    # listing features
    x = init_x(part, idx, drop_slr=(role == BYR_PREFIX))

    # thread features
    x_thread = get_x_thread(threads, idx)

    # add time remaining to x_thread
    if outcome == DELAY:
        x_thread[INT_REMAINING] = calculate_remaining(part, idx, role)

    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx, outcome, role, turn))

    # error checking
    for i in range(1, turn + 1):
        k = 'offer' + str(i)
        if k in x:
            # error checking
            if i == 1:
                check_zero(x[k], [DAYS, DELAY])
            if i % 2 == 1:
                check_zero(x[k], [AUTO, EXP, REJECT])
            if i == 7:
                check_zero(x[k], [MSG])

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--outcome', type=str)
    parser.add_argument('--turn', type=int)
    args = parser.parse_args()
    part, outcome, turn = args.part, args.outcome, args.turn

    # error checking
    assert part in PARTITIONS
    assert outcome in [DELAY, CON, MSG]
    if outcome == DELAY:
        assert turn in range(2, 8)
    elif outcome == MSG:
        assert turn in range(1, 7)
    else:
        assert turn in range(1, 8)

    # model name
    name = '{}{}'.format(outcome, turn)
    print('{}/{}'.format(part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, outcome, turn)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
