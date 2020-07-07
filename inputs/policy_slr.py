import pandas as pd
from inputs.util import save_files, get_x_thread, \
    get_x_offer_init, get_init_data
from utils import init_x, input_partition
from constants import IDX, SLR, CON_MULTIPLIER, TRAIN_MODELS, \
    VALIDATION, TEST, POLICY_SLR
from featnames import EXP, CON, DELAY, AUTO


def construct_x(part=None, idx=None, data=None):
    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(data['threads'], idx,
                            turn_indicators=True)

    # concatenate with the lstg grouping
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(data['offers'], idx, role=SLR))

    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0

    return x


def get_y(df):
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # expired offer is last index
    y[df[EXP]] = CON_MULTIPLIER + 1
    return y


def create_index(offers):
    slr_turn = offers.index.isin(IDX[SLR], level='index')
    censored = offers[EXP] & (offers[DELAY] < 1)
    mask = slr_turn & ~offers[AUTO] & ~censored
    idx = offers[mask].index
    return idx


def process_inputs(part):
    # load dataframes
    data = get_init_data(part)

    # master index
    idx = create_index(data['offers'])

    # outcome and master index
    y = get_y(data['offers'].loc[idx, [CON, EXP]])

    # input features dictionary
    x = construct_x(part=part, idx=y.index, data=data)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_SLR))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_SLR)


if __name__ == '__main__':
    main()
