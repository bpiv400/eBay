import numpy as np
import pandas as pd
from inputs.util import save_files, construct_x_slr, \
    get_sale_norm, create_index_slr
from inputs.policy_slr import input_parameters
from utils import get_cut, slr_reward, max_slr_reward, load_file
from inputs.const import DELTA_MONTH, DELTA_ACTION, C_ACTION
from constants import MONTH, TRAIN_RL, VALIDATION, TEST, \
    NO_ARRIVAL_CUTOFF
from featnames import META, START_PRICE, START_TIME, CON, \
    NO_ARRIVAL


def get_duration(idx, clock, lookup, delay):
    # calculate number of months from listing start to action
    if delay:
        t_now = clock.groupby(clock.index.names[:-1]).shift()
        t_now = t_now.dropna().astype(clock.dtype)
        t_now = t_now.loc[idx]
    else:
        t_now = clock.loc[idx]
    t_start = lookup.start_time.reindex(index=idx, level='lstg')
    months = (t_now - t_start) / MONTH \
        + t_start.index.get_level_values(level='sim')
    return months.rename('months')


def get_sales(offers, clock, lookup):
    norm = get_sale_norm(offers)
    # number of relistings
    relist_count = norm.index.get_level_values(level='sim')
    norm = norm.reset_index(['sim', 'thread', 'index'], drop=True)
    # time of sale
    sale_time = clock[offers[CON] == 1].reset_index(
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


def get_sim_data(part):
    # indices of listings
    p0 = load_file(part, NO_ARRIVAL)
    idx = p0[p0 <= NO_ARRIVAL_CUTOFF].index
    # load (simulated) data
    lookup = load_file(part, 'lookup')
    threads = load_file(part, 'x_thread_sim')
    offers = load_file(part, 'x_offer_sim')
    clock = load_file(part, 'clock_sim')
    # drop listings with infrequent arrivals
    lookup = lookup.reindex(index=idx)
    threads = threads.reindex(index=idx, level='lstg')
    offers = offers.reindex(index=idx, level='lstg')
    clock = clock.reindex(index=idx, level='lstg')
    return lookup, threads, offers, clock


def process_inputs(part, delay):
    # data
    lookup, threads, offers, clock = get_sim_data(part)

    # master index
    idx = create_index_slr(offers, delay)

    # value components
    sales = get_sales(offers, clock, lookup)
    months_since_start = get_duration(idx, clock, lookup, delay)
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
    x = construct_x_slr(part=part, delay=delay, idx=y.index,
                        offers=offers, threads=threads)

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
