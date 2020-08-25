import argparse
import numpy as np
import pandas as pd
from inputs.util import save_files, check_zero, get_x_thread, get_ind_x
from utils import get_remaining, load_file
from constants import IDX, DAY, SIM_PARTITIONS, \
    CON_MULTIPLIER, MAX_DELAY_TURN, INTERVAL_TURN, INTERVAL_CT_TURN
from featnames import CON, NORM, SPLIT, MSG, AUTO, EXP, REJECT, DAYS, \
    DELAY, INT_REMAINING, TIME_FEATS, LOOKUP, SLR, BYR


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
    if turn in IDX[BYR]:
        mask = (df[CON] > 0) & (df[CON] < 1)
    # for sellers, drop accepts, expires, and auto responses
    else:
        mask = ~df[EXP] & ~df[AUTO] & (df[CON] < 1)
    return df.loc[mask, MSG]


def calculate_remaining(clock=None, lstg_start=None, idx=None, turn=None):
    # load timestamps
    lstg_start = lstg_start.reindex(index=idx, level='lstg')
    delay_start = clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')

    # remaining feature
    turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
    remaining = get_remaining(lstg_start, turn_start)

    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)

    return remaining


def get_x_offer(offers=None, idx=None, outcome=None, turn=None):
    # initialize dictionary of offer features
    x_offer = {}
    # dataframe of offer features for relevant threads
    offers = pd.DataFrame(index=idx).join(offers)
    # drop time feats from buyer models
    if turn in IDX[BYR]:
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
            if i > 1 and turn in IDX[SLR]:
                censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
                assert (offer.loc[censored, TIME_FEATS] == 0.0).all().all()
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')

    # error checking
    check_zero(x_offer)

    return x_offer


def get_y_delay(df):
    # convert to seconds
    delay = np.round(df[DAYS] * DAY).astype('int64')
    # drop zero delays
    delay = delay[delay > 0]
    df = df.reindex(index=delay.index)
    # error checking

    assert delay.max() <= MAX_DELAY_TURN
    # convert to periods
    delay //= INTERVAL_TURN
    # replace expired delays with last index
    assert np.all(df.loc[df[DELAY] == 1, EXP])
    delay.loc[delay == INTERVAL_CT_TURN] = -1
    # replace censored delays with negative index
    delay.loc[df[EXP] & (df[DELAY] < 1)] -= INTERVAL_CT_TURN
    return delay


def process_inputs(part, outcome, turn):
    threads = load_file(part, 'x_thread')
    offers = load_file(part, 'x_offer')
    clock = load_file(part, 'clock')
    lookup = load_file(part, LOOKUP)
    lstg_start = lookup.start_time

    # subset to turn
    df = offers.xs(turn, level='index')

    # y and master index
    if outcome == CON:
        y = get_y_con(df, turn)
    elif outcome == MSG:
        y = get_y_msg(df, turn)
    else:
        y = get_y_delay(df)
    idx = y.index

    # thread features
    x = {'thread': get_x_thread(threads, idx)}

    # add time remaining to x_thread
    if outcome == DELAY:
        x['thread'][INT_REMAINING] = calculate_remaining(clock=clock,
                                                         lstg_start=lstg_start,
                                                         idx=idx,
                                                         turn=turn)

    # offer features
    x.update(get_x_offer(offers, idx, outcome, turn))

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=SIM_PARTITIONS)
    parser.add_argument('--outcome', type=str, choices=[DELAY, CON, MSG])
    parser.add_argument('--turn', type=int, choices=range(1, 8))
    args = parser.parse_args()
    part, outcome, turn = args.part, args.outcome, args.turn

    # model name
    name = '{}{}'.format(outcome, turn)
    print('{}/{}'.format(part, name))

    # error checking
    if outcome == DELAY:
        assert turn in range(2, 8)
    elif outcome == MSG:
        assert turn in range(1, 7)

    # input dataframes, output processed dataframes
    d = process_inputs(part, outcome, turn)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
