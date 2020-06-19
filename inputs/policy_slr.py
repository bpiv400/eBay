import argparse
from inputs.util import save_files, construct_x_slr, \
    create_index_slr, get_init_data
from constants import SLR, CON_MULTIPLIER, TRAIN_RL, \
    VALIDATION, TEST
from featnames import EXP, CON


def get_y(df, delay):
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # when choosing delay, expired offer is last index
    if delay:
        y[df[EXP]] = CON_MULTIPLIER + 1
    return y


def process_inputs(part, delay):
    # load dataframes
    data = get_init_data(part)

    # master index
    idx = create_index_slr(data['offers'], delay)

    # outcome and master index
    y = get_y(data['offers'].loc[idx, [CON, EXP]], delay)

    # input features dictionary
    x = construct_x_slr(part=part, delay=delay,
                        idx=y.index, data=data)

    return {'y': y, 'x': x}


def input_parameters(outcome):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, delay = args.part, args.delay
    name = '{}_{}'.format(outcome, SLR)
    if delay:
        name += '_delay'
    print('%s/%s' % (part, name))
    return part, delay, name


def main():
    # extract parameters from command line
    part, delay, name = input_parameters('policy')

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_RL, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
