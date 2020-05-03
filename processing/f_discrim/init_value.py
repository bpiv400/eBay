import numpy as np
import pandas as pd
from processing.processing_utils import init_x, get_x_thread, load_file
from processing.e_inputs.inputs_utils import save_files
from processing.e_inputs.init_policy import calculate_remaining, \
    add_turn_indicators, get_x_offer, input_parameters
from processing.processing_consts import MONTHLY_DISCOUNT
from constants import IDX, BYR_PREFIX, SLR_PREFIX, MONTH, \
    LISTING_FEE, NO_ARRIVAL_CUTOFF, MAX_NORM_VALUE
from featnames import INT_REMAINING, EXP, AUTO, CON, NORM, \
    META, START_PRICE, START_TIME
from utils import get_cut


def get_discounted_values(sales, months):
    # join sales and timestamps
    df = months.to_frame().join(sales)
    # time difference
    td = df.months_to_sale - df.months
    assert np.all(td >= 0)
    # discounted listing fees
    M = np.ceil(df.months_to_sale) - np.ceil(df.months)
    k = df.months % 1
    delta = MONTHLY_DISCOUNT ** k
    delta *= 1 - MONTHLY_DISCOUNT ** (M+1)
    delta /= 1 - MONTHLY_DISCOUNT
    costs = LISTING_FEE * delta
    # discounted sale price
    proceeds = df.gross * (MONTHLY_DISCOUNT ** td)
    return proceeds - costs


def get_duration(offers, clock, lookup, role):
    mask = offers.index.isin(IDX[role], level='index') \
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
    # gross price
    cut = lookup[META].apply(get_cut)
    gross = norm * (1-cut) * lookup[START_PRICE]
    return pd.concat([gross.rename('gross'),
                      months.rename('months_to_sale')], axis=1)


def get_data(part):
    # lookup file
    lookup = load_file(part, 'lookup')
    # load simulated data
    threads = load_file(part, 'x_thread_sim')
    offers = load_file(part, 'x_offer_sim')
    clock = load_file(part, 'clock_sim')
    # drop listings with infrequent arrivals
    lookup = lookup.loc[lookup.p_no_arrival < NO_ARRIVAL_CUTOFF, :]
    threads = threads.reindex(index=lookup.index, level='lstg')
    offers = offers.reindex(index=lookup.index, level='lstg')
    clock = clock.reindex(index=lookup.index, level='lstg')
    return lookup, threads, offers, clock


def process_inputs(part, role, delay):
    # data
    lookup, threads, offers, clock = get_data(part)

    # value components
    sales = get_sales(offers, clock, lookup)
    months = get_duration(offers, clock, lookup, role)
    idx = months.index
    values = get_discounted_values(sales, months)
    norm_values = values / lookup[START_PRICE].reindex(
        index=idx, level='lstg')
    y = np.maximum(np.round(norm_values * 100), 0.).astype('uint8')
    assert (y.max() <= MAX_NORM_VALUE) and (y.min() >= 0)

    # listing features
    x = init_x(part, idx)

    # remove auto accept/reject features from x['lstg'] for buyer models
    if role == BYR_PREFIX:
        x['lstg'].drop(['auto_decline', 'auto_accept',
                        'has_decline', 'has_accept'],
                       axis=1, inplace=True)

    # thread features
    x_thread = get_x_thread(threads, idx)
    if delay:
        x_thread[INT_REMAINING] = calculate_remaining(lstg_start=lookup.start_time,
                                                      clock=clock,
                                                      idx=idx)
    x_thread = add_turn_indicators(x_thread)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx, role))

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part, role, delay = input_parameters()
    name = 'init_value_{}'.format(role)
    print('%s/%s' % (part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, role, delay)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
