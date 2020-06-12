import argparse
from inputs.utils import get_policy_data, save_files, construct_x_init
from constants import IDX, SLR_PREFIX, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST
from featnames import AUTO, EXP, CON


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


def process_inputs(part, delay):
    # load dataframes
    offers, threads, clock, lstg_start = get_policy_data(part)

    # restrict by role, drop auto replies
    role_mask = offers.index.isin(IDX[SLR_PREFIX], level='index')
    df = offers[~offers[AUTO] & role_mask]

    # outcome and master index
    y = get_y(df, delay)

    # input features dictionary

    x = construct_x_init(part=part, role=SLR_PREFIX, delay=delay,
                         idx=y.index, offers=offers,
                         threads=threads, clock=clock,
                         lstg_start=lstg_start)
    return {'y': y, 'x': x}


def input_parameters(outcome):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, delay = args.part, args.delay
    name = '{}_{}'.format(outcome, SLR_PREFIX)
    if delay:
        name += '_delay'
    print('%s/%s' % (part, name))
    return part, delay, name


def main():
    # extract parameters from command line
    part, delay, name = input_parameters('policy')

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
