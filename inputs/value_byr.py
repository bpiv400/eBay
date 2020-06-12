import numpy as np
from inputs.utils import save_files, get_value_data, \
    construct_x_init, get_sale_norm
from utils import byr_reward, input_partition
from inputs.constants import DELTA_MONTH, DELTA_ACTION, C_ACTION
from constants import IDX, BYR_PREFIX, MONTH, \
    TRAIN_RL, VALIDATION, TEST
from featnames import EXP, AUTO, START_PRICE, START_TIME


def get_duration(offers, clock, lookup):
    mask = offers.index.isin(IDX[SLR_PREFIX], level='index') \
           & ~offers[EXP] & ~offers[AUTO]
    idx = mask[mask].index
    elapsed = (clock.loc[mask] - lookup.start_time.reindex(
        index=idx, level='lstg')) / MONTH
    months = elapsed + elapsed.index.get_level_values(level='sim')
    return months.rename('months')


def process_inputs(part):
    # data
    lookup, threads, offers, clock = get_value_data(part)

    # value components
    norm = get_sale_norm(offers)
    net_value = (1 - norm) * lookup[START_PRICE]

    sales = get_sales(offers, clock, lookup)
    months_since_start = get_duration(offers, clock, lookup)
    df = months_since_start.to_frame().join(sales)

    # discounted values
    values = byr_reward(net_value=df.net_value,
                        months_diff=df.months_diff,
                        monthly_discount=DELTA_MONTH)

    # normalize by start_price
    norm_values = values / lookup[START_PRICE].reindex(
        index=values.index, level='lstg')
    assert norm_values.max() <= 1

    # integer values
    y = np.maximum(np.round(norm_values * 100), 0.).astype('uint8')

    # input feature dictionary
    x = construct_x_init(part=part, role=BYR_PREFIX, delay=True,
                         idx=y.index, offers=offers,
                         threads=threads, clock=clock,
                         lstg_start=lookup[START_TIME])

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'value_{}_delay'.format(BYR_PREFIX)
    print('%s/%s' % (part, name))

    # policy is trained using TRAIN_RL
    assert part in [TRAIN_RL, VALIDATION, TEST]

    # action discount and cost not implemented
    if DELTA_ACTION != 1 or C_ACTION != 0:
        raise NotImplementedError()

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
