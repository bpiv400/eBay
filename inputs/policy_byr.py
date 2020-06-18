import pandas as pd
from inputs.util import save_files, construct_x_byr, \
    create_index_byr, get_init_data
from utils import input_partition
from constants import BYR, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST
from featnames import CON, START_TIME


def get_y(idx, idx1, offers):
    y = pd.Series(0, dtype='int8', index=idx)
    y.loc[idx1] = (offers[CON] * CON_MULTIPLIER).astype(y.dtype)
    return y


def process_inputs(part):
    # load dataframes
    data = get_init_data(part)

    # master index
    idx, idx1 = create_index_byr(data['clock'],
                                 data['lookup'][START_TIME])

    # outcome
    y = get_y(idx, idx1, data['offers'])

    # input feature dictionary
    x = construct_x_byr(part=part, idx=y.index, data=data)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'policy_{}_delay'.format(BYR)
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
