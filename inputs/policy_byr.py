import pandas as pd
from inputs.util import save_files, get_x_thread, get_x_offer_init, get_ind_x
from utils import input_partition, load_file
from constants import IDX, CON_MULTIPLIER, POLICY_BYR
from featnames import CON, EXP, THREAD_COUNT, LOOKUP, BYR, X_OFFER, X_THREAD, CLOCK


def construct_x(idx=None, threads=None, offers=None):
    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)
    # initialize x with thread features
    x = {'thread': x_thread}
    # offer features
    x.update(get_x_offer_init(offers, idx, role=BYR))
    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0
    return x


def get_y(idx, idx1, offers):
    y = pd.Series(0, dtype='int8', index=idx)
    y.loc[idx1] = (offers[CON] * CON_MULTIPLIER).astype(y.dtype)
    return y


def create_index(clock=None, offers=None):
    # buyer turns
    s = clock[clock.index.isin(IDX[BYR], level='index')]
    # remove expirations
    s = s[~offers[EXP]]
    return s.index


def process_inputs(part):
    # load dataframes
    offers = load_file(part, X_OFFER)
    threads = load_file(part, X_THREAD)
    clock = load_file(part, CLOCK)
    lstgs = load_file(part, LOOKUP).index

    # master index
    idx = create_index(clock=clock, offers=offers)

    # outcome
    y = (offers.loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input feature dictionary
    x = construct_x(idx=idx, threads=threads, offers=offers)

    # indices for listing features
    idx_x = get_ind_x(lstgs=lstgs, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_BYR))

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_BYR)


if __name__ == '__main__':
    main()
