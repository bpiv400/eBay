import argparse
import pandas as pd
from inputs.inputs_utils import save_files, get_x_thread, \
    get_x_offer_init, calculate_remaining
from utils import load_file, init_x
from constants import IDX, SLR_PREFIX, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST
from featnames import AUTO, EXP, CON, INT_REMAINING, START_TIME


def get_y(df, delay):
    # when only choosing concession, drop expirations
    if not delay:
        df = df[~df[EXP]]
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # when choosing delay, expired offer is last index
    if delay:
        y[df[EXP]] = CON_MULTIPLIER + 1
    return y


# loads data and calls helper functions to construct train inputs
def process_inputs(part, delay):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')

    # restrict by role, drop auto replies
    role_mask = offers.index.isin(IDX[SLR_PREFIX], level='index')
    df = offers[~offers[AUTO] & role_mask]

    # outcome and master index
    y = get_y(df, delay)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)
    if delay:
        lstg_start = load_file(part, 'lookup')[START_TIME]
        clock = load_file(part, 'clock')
        x_thread[INT_REMAINING] = \
            calculate_remaining(lstg_start=lstg_start,
                                clock=clock,
                                idx=idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(offers, idx, role=SLR_PREFIX, delay=delay))

    return {'y': y, 'x': x}


def input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, delay = args.part, args.delay
    return part, delay


def main():
    # extract parameters from command line
    part, delay = input_parameters()
    name = 'policy_{}'.format(SLR_PREFIX)
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
