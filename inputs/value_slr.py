import numpy as np
import pandas as pd
from inputs.util import save_files, get_value_data, \
    construct_x_init, get_sale_norm
from inputs.policy_slr import input_parameters
from utils import get_cut, slr_reward, max_slr_reward
from inputs.const import DELTA_MONTH, DELTA_ACTION, C_ACTION
from constants import IDX, SLR_PREFIX, MONTH, TRAIN_RL, VALIDATION, TEST
from featnames import EXP, AUTO, META, START_PRICE, START_TIME


def get_duration(offers, clock, lookup):
    mask = offers.index.isin(IDX[SLR_PREFIX], level='index') \
           & ~offers[EXP] & ~offers[AUTO]
    idx = mask[mask].index
    elapsed = (clock.loc[mask] - lookup.start_time.reindex(
        index=idx, level='lstg')) / MONTH
    months = elapsed + elapsed.index.get_level_values(level='sim')
    return months.rename('months')


def get_sales(offers, clock, lookup):
    norm = get_sale_norm(offers)
    # number of relistings
    relist_count = norm.index.get_level_values(level='sim')
    norm = norm.reset_index(['sim', 'thread', 'index'], drop=True)
    # time of sale
    sale_time = clock[is_sale].reset_index(
        ['sim', 'thread', 'index'], drop=True)
    elapsed = (sale_time - lookup[START_TIME]) / MONTH
    assert (elapsed >= 0).all()
    months = relist_count + elapsed
    # gross bin_price and gross actual price
    cut = lookup[META].apply(get_cut)
    bin_gross = (1-cut) * lookup[START_PRICE]
    gross = norm * bin_gross
    return pd.concat([bin_gross.rename('bin_proceeds'),
                      gross.rename('sale_proceeds'),
                      months.rename('months_to_sale')], axis=1)


def process_inputs(part, delay):
    # data
    lookup, threads, offers, clock = get_value_data(part)

    # value components
    sales = get_sales(offers, clock, lookup)
    months_since_start = get_duration(offers, clock, lookup)
    df = months_since_start.to_frame().join(sales)

    # discounted values
    values = slr_reward(months_to_sale=df.months_to_sale,
                        months_since_start=df.months,
                        sale_proceeds=df.sale_proceeds,
                        monthly_discount=DELTA_MONTH)

    max_values = max_slr_reward(months_since_start=df.months,
                                bin_proceeds=df.bin_proceeds,
                                monthly_discount=DELTA_MONTH)

    # normalize by max values
    norm_values = values / max_values
    assert norm_values.max() <= 1

    # integer values
    y = np.maximum(np.round(norm_values * 100), 0.).astype('uint8')

    # input features dictionary
    x = construct_x_init(part=part, role=SLR_PREFIX, delay=delay,
                         idx=y.index, offers=offers,
                         threads=threads, clock=clock,
                         lstg_start=lookup[START_TIME])

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part, delay, name = input_parameters('value')

    # policy is trained using TRAIN_RL
    assert part in [TRAIN_RL, VALIDATION, TEST]

    # action discount and cost not implemented
    if DELTA_ACTION != 1 or C_ACTION != 0:
        raise NotImplementedError()

    # input dataframes, output processed dataframes
    d = process_inputs(part, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
