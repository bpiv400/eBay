import argparse
import numpy as np
import pandas as pd
from inputs.util import save_files, get_x_thread, get_ind_x, check_zero
from utils import load_data, get_role
from constants import IDX, CON_MULTIPLIER
from featnames import CON, AUTO, LOOKUP, SLR, INDEX, MSG, REJECT, NORM, \
    COMMON, THREAD, X_THREAD, X_OFFER, CLOCK, THREAD_COUNT, TIME_FEATS, VALIDATION


def get_x_offer(offers=None, idx=None, byr=False):
    # restrict to relevant threads
    threads = idx.droplevel(level=INDEX).unique()
    df = pd.DataFrame(index=threads).join(offers)

    # drop time feats for byr
    if byr:
        df = offers.drop(TIME_FEATS, axis=1)

    # turn features
    turns = np.sort(idx.get_level_values(INDEX).unique()).astype(np.int8)
    x_offer = {}
    for t in range(1, int(max(turns)) + 1):
        # offer features at turn i, and turn number
        offer = df.xs(t, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
        turn = offer.index.get_level_values(level=INDEX)

        # msg is 0 for turns of focal player
        if t in turns:
            offer.loc[:, MSG] = 0.

        # all features are zero for future turns
        offer.loc[t > turn, :] = 0.

        # for current turn, post-delay features set to 0
        offer.loc[t == turn, [CON, REJECT, NORM, COMMON]] = 0.
        if t in turns:
            assert offer.loc[t == turn, [AUTO, MSG]].max().max() == 0.

        # put in dictionary
        x_offer['offer{}'.format(t)] = offer

    # error checking
    check_zero(x_offer, byr_exp=(not byr))

    return x_offer


def construct_x(idx=None, data=None, byr=False):
    # initialize dictionary with thread features
    x = {THREAD: get_x_thread(data[X_THREAD], idx, turn_indicators=True)}
    if byr:
        x[THREAD].drop(THREAD_COUNT, axis=1, inplace=True)

    # offer features
    x.update(get_x_offer(offers=data[X_OFFER], idx=idx, byr=byr))

    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0

    return x


def create_index(data=None, byr=False):
    turns = [1, 3, 5] if byr else IDX[SLR]
    mask = pd.Series(data[CLOCK].index.isin(turns, level=INDEX),
                     index=data[CLOCK].index)

    # drop auto-rejects
    if not byr:
        mask = mask & ~data[X_OFFER][AUTO]

    idx = mask[mask].index
    return idx


def process_inputs(data=None, byr=False):
    # master index
    idx = create_index(data=data, byr=byr)

    # outcome and master index
    y = (data[X_OFFER].loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input features dictionary
    x = construct_x(idx=idx, data=data, byr=byr)

    # indices for fixed features
    idx_x = get_ind_x(lstgs=data[LOOKUP].index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    byr = parser.parse_args().byr

    role = get_role(byr)
    print('{}/{}'.format(VALIDATION, get_role(byr)))

    # input dataframes, output processed dataframes
    data = load_data(part=VALIDATION, clock=True)
    d = process_inputs(data=data, byr=byr)

    # save various output files
    save_files(d, VALIDATION, role)


if __name__ == '__main__':
    main()
