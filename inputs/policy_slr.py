import pandas as pd
from inputs.util import save_files, get_x_thread, get_ind_x, check_zero
from utils import input_partition, load_file
from constants import IDX, CON_MULTIPLIER, POLICY_SLR
from featnames import EXP, CON, DELAY, AUTO, LOOKUP, SLR, INDEX, MSG, \
    REJECT, NORM, SPLIT, THREAD, X_THREAD, X_OFFER


def get_x_offer(offers, idx):
    # initialize dictionary of offer features
    x_offer = {}

    # dataframe of offer features for relevant threads
    threads = idx.droplevel(level=INDEX).unique()
    offers = pd.DataFrame(index=threads).join(offers)

    # turn features
    for i in range(1, max(IDX[SLR]) + 1):
        # offer features at turn i, and turn number
        offer = offers.xs(i, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
        turn = offer.index.get_level_values(level=INDEX)

        # msg is 0 for turns of focal player
        if i in IDX[SLR]:
            offer.loc[:, MSG] = 0.

        # all features are zero for future turns
        offer.loc[i > turn, :] = 0.

        # for current turn, post-delay features set to 0
        offer.loc[i == turn, [CON, REJECT, NORM, SPLIT]] = 0.
        if i in IDX[SLR]:
            assert offer.loc[i == turn, [AUTO, MSG]].max().max() == 0.

        # put in dictionary
        x_offer['offer{}'.format(i)] = offer

    # error checking
    check_zero(x_offer)

    return x_offer


def construct_x(idx=None, threads=None, offers=None):
    # initialize dictionary with thread features
    x = {THREAD: get_x_thread(threads, idx, turn_indicators=True)}
    # offer features
    x.update(get_x_offer(offers, idx))
    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0
    return x


def create_index(offers):
    slr_turn = offers.index.isin(IDX[SLR], level=INDEX)
    censored = offers[EXP] & (offers[DELAY] < 1)
    mask = slr_turn & ~offers[AUTO] & ~censored
    idx = mask[mask].index
    return idx


def process_inputs(part):
    # load dataframes
    lstgs = load_file(part, LOOKUP).index
    threads = load_file(part, X_THREAD)
    offers = load_file(part, X_OFFER)

    # master index
    idx = create_index(offers)

    # outcome and master index
    y = (offers.loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input features dictionary
    x = construct_x(idx=idx, threads=threads, offers=offers)

    # indices for fixed features
    idx_x = get_ind_x(lstgs=lstgs, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_SLR))

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_SLR)


if __name__ == '__main__':
    main()
