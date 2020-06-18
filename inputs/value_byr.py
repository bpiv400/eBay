import numpy as np
import pandas as pd
from inputs.util import save_files, construct_x_byr, \
    get_sale_norm, create_index_byr
from utils import byr_reward, input_partition, load_file
from inputs.const import DELTA_MONTH, DELTA_ACTION, C_ACTION
from constants import BYR, TRAIN_RL, VALIDATION, TEST, DAY, MONTH
from featnames import START_PRICE, START_TIME, CON


def get_months(idx, clock, offers, lstg_start):
    # months is inf if no sale
    months = pd.Series(np.inf, index=idx, name='months_diff')
    # split index by turn
    idx_first = idx[idx.isin([1], level='index')]
    idx_next = idx[idx.isin([3, 5, 7], level='index')]
    # beginning of delay window for first offer
    t_first = lstg_start.reindex(index=idx_first, level='lstg')
    t_first += DAY * t_first.index.get_level_values(level='day')
    # beginning of delay window for subsequent offers
    t_next = clock.groupby(clock.index.names[:-1]).shift()
    t_next = t_next.dropna().astype(clock.dtype)
    t_next = t_next.to_frame().assign(day=0).set_index(
        'day', append=True).squeeze().loc[idx_next]
    # for sales, fill in with months to sale
    t_sale = clock[offers[CON] == 1].reset_index(
        'index', drop=True).rename('sale_time')
    t_now = pd.concat([t_first, t_next]).rename('delay_start')
    df = t_now.to_frame().join(t_sale).reorder_levels(
        t_now.index.names)
    df = df[~df.sale_time.isna()].astype(clock.dtype)
    months.loc[df.index] = (df.sale_time - df.delay_start) / MONTH
    return months


def process_inputs(part):
    # data
    lookup = load_file(part, 'lookup')
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    clock = load_file(part, 'clock')

    # master index
    idx, _ = create_index_byr(clock, lookup[START_TIME])

    # value components
    norm = get_sale_norm(offers).reset_index('index', drop=True)
    net = ((1 - norm) * lookup[START_PRICE]).rename('net_value')
    months = get_months(idx, clock, offers, lookup[START_TIME])
    df = months.to_frame().join(net)
    df.index = df.index.reorder_levels(months.index.names)

    # discounted values
    values = byr_reward(net_value=df.net_value,
                        months_diff=df.months_diff,
                        monthly_discount=DELTA_MONTH)
    values[values.isna()] = 0.

    # normalize by start_price
    norm_values = values / lookup[START_PRICE].reindex(
        index=values.index, level='lstg')
    assert norm_values.max() <= 1

    # integer values
    y = np.maximum(np.round(norm_values * 100), 0.).astype('uint8')

    # input feature dictionary
    x = construct_x_byr(part=part, idx=y.index, offers=offers,
                        threads=threads, clock=clock,
                        lstg_start=lookup[START_TIME])

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'value_{}_delay'.format(BYR)
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
