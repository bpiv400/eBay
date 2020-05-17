import numpy as np
import pandas as pd
from inputs.inputs_utils import save_files, get_x_thread, \
    calculate_remaining, get_x_offer_init
from inputs.policy_slr import input_parameters
from utils import load_file, init_x, get_cut, slr_reward, max_slr_reward
from inputs.inputs_consts import DELTA_MONTH
from constants import IDX, SLR_PREFIX, MONTH, NO_ARRIVAL_CUTOFF, \
    TRAIN_RL, VALIDATION, TEST
from featnames import INT_REMAINING, EXP, AUTO, CON, NORM, \
    META, START_PRICE, START_TIME, NO_ARRIVAL


def get_duration(offers, clock, lookup):
    mask = offers.index.isin(IDX[SLR_PREFIX], level='index') \
           & ~offers[EXP] & ~offers[AUTO]
    idx = mask[mask].index
    elapsed = (clock.loc[mask] - lookup.start_time.reindex(
        index=idx, level='lstg')) / MONTH
    months = elapsed + elapsed.index.get_level_values(level='sim')
    return months.rename('months')


def get_sales(offers, clock, lookup):
    # normalized sale price
    is_sale = offers[CON] == 1.
    norm = offers.loc[is_sale, NORM]
    slr_turn = norm.index.isin(IDX[SLR_PREFIX], level='index')
    norm_slr = 1 - norm.loc[slr_turn]
    norm.loc[slr_turn] = norm_slr
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


def get_data(part):
    # lookup file
    lookup = load_file(part, 'lookup')
    # load simulated data
    threads = load_file(part, 'x_thread_sim')
    offers = load_file(part, 'x_offer_sim')
    clock = load_file(part, 'clock_sim')
    # drop listings with infrequent arrivals
    lookup = lookup.loc[lookup[NO_ARRIVAL] < NO_ARRIVAL_CUTOFF, :]
    threads = threads.reindex(index=lookup.index, level='lstg')
    offers = offers.reindex(index=lookup.index, level='lstg')
    clock = clock.reindex(index=lookup.index, level='lstg')
    return lookup, threads, offers, clock


def process_inputs(part, delay):
    # data
    lookup, threads, offers, clock = get_data(part)

    # value components
    sales = get_sales(offers, clock, lookup)
    months_since_start = get_duration(offers, clock, lookup)
    df = months_since_start.to_frame().join(sales)
    idx = df.index

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

    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)
    if delay:
        x_thread[INT_REMAINING] = calculate_remaining(lstg_start=lookup.start_time,
                                                      clock=clock,
                                                      idx=idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(offers, idx, role=SLR_PREFIX, delay=delay))

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part, delay = input_parameters()
    name = 'value_{}'.format(SLR_PREFIX)
    if delay:
        name += '_delay'
    print('%s/%s' % (part, name))

    # policy is trained using TRAIN_RL
    assert part in [TRAIN_RL, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
