import argparse
import numpy as np
import pandas as pd
from processing.util import get_days_delay
from inputs.util import save_files, check_zero, get_x_thread, get_ind_x
from utils import get_remaining, load_file
from constants import IDX, MAX_DELAY_TURN, \
    INTERVAL_TURN, INTERVAL_CT_TURN
from featnames import INT_REMAINING, TIME_FEATS, LOOKUP, SLR, BYR, DELAY, \
    X_THREAD, X_OFFER, CLOCK, LSTG, THREAD, INDEX, START_TIME, END_TIME, CON, SIM_PARTITIONS


def calculate_remaining(clock=None, lstg_start=None, idx=None, turn=None):
    # load timestamps
    lstg_start = lstg_start.reindex(index=idx, level=LSTG)
    turn_start = clock.xs(turn - 1, level=INDEX).reindex(index=idx)
    # remaining feature
    remaining = get_remaining(lstg_start, turn_start)
    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)
    return remaining


def get_x_offer(offers=None, idx=None, turn=None):
    # dataframe of offer features for relevant threads
    offers = pd.DataFrame(index=idx).join(offers)
    # drop time feats from buyer models
    if turn in IDX[BYR]:
        offers.drop(TIME_FEATS, axis=1, inplace=True)
    # turn features
    d = {}
    for i in range(1, turn):
        k = 'offer{}'.format(i)
        d[k] = offers.xs(i, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
    # error checking
    check_zero(d)
    return d


def get_y(offers=None, clock=None, lstg_end=None, turn=None):
    # identify finished threads
    con = offers[CON].sort_index().reset_index(level=INDEX)
    last = con.groupby([LSTG, THREAD]).last()
    done = (last[CON] == 1) | ((last[CON] == 0) & (last[INDEX] % 2 == 1))
    new = last.loc[~done, INDEX] + 1
    assert new.min() > 1
    assert new.max() <= 7
    idx = pd.MultiIndex.from_frame(new.reset_index())
    # censored timestamps
    ts_cens = lstg_end.reindex(index=idx, level=LSTG)
    ts = pd.concat([clock, ts_cens], axis=0).sort_index()
    # delay in seconds
    _, delay = get_days_delay(ts.unstack())
    delay_seconds = np.round(delay * MAX_DELAY_TURN).astype('int64')
    # convert to periods
    intervals = delay_seconds // INTERVAL_TURN
    # replace expired delays with last index
    intervals.loc[intervals == INTERVAL_CT_TURN] = -1
    # replace censored delays with negative index
    assert intervals.loc[idx].min() >= 0
    intervals.loc[idx] -= INTERVAL_CT_TURN
    # drop automatic offers
    if turn in IDX[SLR]:
        intervals = intervals[delay_seconds > 0]
    # restrict to turn
    intervals = intervals.xs(turn, level=INDEX)
    return intervals


def process_inputs(part, turn):
    threads = load_file(part, X_THREAD)
    offers = load_file(part, X_OFFER)
    clock = load_file(part, CLOCK)
    lookup = load_file(part, LOOKUP)

    # y and master index
    y = get_y(offers=offers,
              clock=clock,
              lstg_end=lookup[END_TIME],
              turn=turn)
    idx = y.index

    # thread features
    x = {THREAD: get_x_thread(threads, idx)}

    # add time remaining to x_thread
    x[THREAD][INT_REMAINING] = calculate_remaining(clock=clock,
                                                   lstg_start=lookup[START_TIME],
                                                   idx=idx,
                                                   turn=turn)

    # offer features
    x.update(get_x_offer(offers, idx, turn))

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=SIM_PARTITIONS)
    parser.add_argument('--turn', type=int, choices=range(2, 8))
    args = parser.parse_args()
    part, turn = args.part, args.turn

    # model name
    name = '{}{}'.format(DELAY, turn)
    print('{}/{}'.format(part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, turn)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
