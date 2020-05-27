import pandas as pd
from inputs.inputs_utils import get_policy_data, save_files, construct_x_init
from utils import input_partition
from constants import IDX, BYR_PREFIX, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST, DAY
from featnames import CON


def get_y(df, lstg_start):
    # waiting periods before first offer
    arrival_time = df.clock.xs(1, level='index', drop_level=False)
    days = ((arrival_time - lstg_start) // DAY).rename('day')
    # expand index
    days0 = days[days > 0] - 1
    wide = days0.to_frame().assign(con=0).set_index(
        'day', append=True).squeeze().unstack()
    for i in wide.columns:
        wide.loc[i < days0, i] = 0.
    y0 = wide.stack().astype('int8')
    # combine concession and waiting time
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    y = y.to_frame().join(days.reindex(
        index=y.index, fill_value=0)).set_index(
        'day', append=True).squeeze()
    y = pd.concat([y0, y]).sort_index()
    return y


def process_inputs(part):
    # load dataframes
    offers, threads, clock, lstg_start = get_policy_data(part)

    # restict by role, join timestamps and offer features
    df = clock[clock.index.isin(IDX[BYR_PREFIX], level='index')]
    df = df.to_frame().join(offers)

    # outcome
    y = get_y(df, lstg_start)

    # input feature dictionary
    x = construct_x_init(part=part, role=BYR_PREFIX, delay=True,
                         idx=y.index, offers=offers,
                         threads=threads, clock=clock,
                         lstg_start=lstg_start)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'policy_{}_delay'.format(BYR_PREFIX)
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
