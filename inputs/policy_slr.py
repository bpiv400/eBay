import argparse
import pandas as pd
from inputs.util import save_files, get_x_thread, \
    get_x_offer_init, get_init_data
from utils import init_x
from constants import IDX, SLR, CON_MULTIPLIER, TRAIN_MODELS, \
    VALIDATION, TEST
from featnames import EXP, CON, DELAY, AUTO


def construct_x(part=None, delay=None, idx=None, data=None):
    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(data['threads'], idx,
                            turn_indicators=True)

    # concatenate with the lstg grouping
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(data['offers'], idx,
                              role=SLR, delay=delay))

    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0

    return x


def get_y(df, delay):
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # when choosing delay, expired offer is last index
    if delay:
        y[df[EXP]] = CON_MULTIPLIER + 1
    return y


def create_index(offers=None, delay=None):
    slr_turn = offers.index.isin(IDX[SLR], level='index')
    censored = offers[EXP] & (offers[DELAY] < 1)
    mask = slr_turn & ~offers[AUTO] & ~censored
    if not delay:  # when not choosing delay, drop expirations
        mask = mask & ~offers[EXP]
    idx = offers[mask].index
    return idx


def process_inputs(part, delay):
    # load dataframes
    data = get_init_data(part)

    # master index
    idx = create_index(offers=data['offers'], delay=delay)

    # outcome and master index
    y = get_y(data['offers'].loc[idx, [CON, EXP]], delay)

    # input features dictionary
    x = construct_x(part=part, delay=delay, idx=y.index, data=data)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, delay = args.part, args.delay
    name = 'policy_{}'.format(SLR)
    if delay:
        name += '_delay'
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
