import argparse
import numpy as np
import pandas as pd
from inputs.utils import save_files, check_zero, get_x_thread
from utils import get_remaining, load_file, init_x
from inputs.constants import INTERVAL, INTERVAL_COUNTS
from constants import IDX, DAY, MAX_DELAY, BYR_PREFIX, SLR_PREFIX, \
    PARTITIONS, CON_MULTIPLIER
from featnames import CON, NORM, SPLIT, MSG, AUTO, EXP, REJECT, DAYS, \
    DELAY, INT_REMAINING, TIME_FEATS


def get_y_con(df, turn):
    # drop zero delay and expired offers
    mask = ~df[AUTO] & ~df[EXP]
    # concession is an int from 0 to 100
    y = (df.loc[mask, CON] * CON_MULTIPLIER).astype('int8')
    # boolean for accept in turn 7
    if turn == 7:
        y = y == 100
    return y


def get_y_msg(df, turn):
    # for buyers, drop accepts and rejects
    if turn in IDX[BYR_PREFIX]:
        mask = (df[CON] > 0) & (df[CON] < 1)
    # for sellers, drop accepts, expires, and auto responses
    else:
        mask = ~df[EXP] & ~df[AUTO] & (df[CON] < 1)
    return df.loc[mask, MSG]


def calculate_remaining(clock, lstg_start, idx, turn):
    # load timestamps
    lstg_start = lstg_start.reindex(index=idx, level='lstg')
    delay_start = clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')

    # remaining feature
    turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
    remaining = get_remaining(lstg_start, turn_start, MAX_DELAY[turn])

    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)

    return remaining


def get_x_offer(offers, idx, outcome, turn):
    # initialize dictionary of offer features
    x_offer = {}
    # dataframe of offer features for relevant threads
    offers = pd.DataFrame(index=idx).join(offers)
    # drop time feats from buyer models
    if turn in IDX[BYR_PREFIX]:
        offers.drop(TIME_FEATS, axis=1, inplace=True)
    # last turn to include
    if outcome == DELAY:
        last = turn - 1
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
        # assert that time feats are all zero for censored observations
        else:
            if i > 1 and turn in IDX[SLR_PREFIX]:
                censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
                assert (offer.loc[censored, TIME_FEATS] == 0.0).all().all()
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')
    return x_offer


def get_y_delay(df, turn):
    # convert to seconds
    delay = np.round(df[DAYS] * DAY).astype('int64')
    # drop zero delays
    delay = delay[delay > 0]
    df = df.reindex(index=delay.index)
    # error checking
    assert delay.max() <= MAX_DELAY[turn]
    # convert to periods
    delay //= INTERVAL[turn]
    # replace expired delays with last index
    assert np.all(df.loc[df[DELAY] == 1, EXP])
    delay.loc[delay == INTERVAL_COUNTS[turn]] = -1
    # replace censored delays with negative index
    delay.loc[df[EXP] & (df[DELAY] < 1)] -= INTERVAL_COUNTS[turn]
    return delay


def process_inputs(part, outcome, turn):
    threads = load_file(part, 'x_thread')
    offers = load_file(part, 'x_offer')
    clock = load_file(part, 'clock')
    lstg_start = load_file(part, 'lookup').start_time

    # subset to turn
    df = offers.xs(turn, level='index')

    # y and master index
    if outcome == CON:
        y = get_y_con(df, turn)
    elif outcome == MSG:
        y = get_y_msg(df, turn)
    else:
        y = get_y_delay(df, turn)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(threads, idx)

    # add time remaining to x_thread
    if outcome == DELAY:
        x_thread[INT_REMAINING] = calculate_remaining(clock,
                                                      lstg_start,
                                                      idx,
                                                      turn)

    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx, outcome, turn))

    # error checking
    check_zero(x)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--outcome', type=str)
    parser.add_argument('--turn', type=int)
    args = parser.parse_args()
    part, outcome, turn = args.part, args.outcome, args.turn

    # model name
    name = '{}{}'.format(outcome, turn)
    print('{}/{}'.format(part, name))

    # error checking
    assert part in PARTITIONS
    assert outcome in [DELAY, CON, MSG]
    if outcome == DELAY:
        assert turn in range(2, 8)
    elif outcome == MSG:
        assert turn in range(1, 7)
    else:
        assert turn in range(1, 8)

    # input dataframes, output processed dataframes
    d = process_inputs(part, outcome, turn)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
