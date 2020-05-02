import numpy as np
import pandas as pd
from processing.processing_utils import init_x, get_x_thread
from processing.e_inputs.inputs_utils import save_files
from processing.e_inputs.init_policy import calculate_remaining, \
    add_turn_indicators, get_x_offer, input_parameters
from processing.f_discrim.discrim_utils import concat_sim_chunks
from processing.processing_consts import MONTHLY_DISCOUNT
from constants import IDX, BYR_PREFIX, SLR_PREFIX, MONTH, LISTING_FEE
from featnames import INT_REMAINING, EXP, AUTO, CON, NORM
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


def get_duration(d, role):
    df = d['offers']
    mask = df.index.isin(IDX[role], level='index') & ~df[EXP] & ~df[AUTO]
    clock = d['clock'].loc[mask]
    elapsed = (clock - d['lookup'].start_time.reindex(
        index=clock.index, level='lstg')) / MONTH
    months = elapsed + elapsed.index.get_level_values(level='sim')
    return months.rename('months')


def get_sales(d):
    # normalized sale price
    is_sale = d['offers'][CON] == 1.
    norm = d['offers'].loc[is_sale, NORM]
    slr_turn = norm.index.isin(IDX[SLR_PREFIX], level='index')
    norm_slr = 1 - norm.loc[slr_turn]
    norm.loc[slr_turn] = norm_slr
    # number of relistings
    relist_count = norm.index.get_level_values(level='sim')
    norm = norm.reset_index(['sim', 'thread', 'index'], drop=True)
    # time of sale
    sale_time = d['clock'][is_sale].reset_index(
        ['sim', 'thread', 'index'], drop=True)
    elapsed = (sale_time - d['lookup'].start_time) / MONTH
    assert (elapsed >= 0).all()
    months = relist_count + elapsed
    # gross price
    cut = d['lookup'].meta.apply(get_cut)
    gross = norm * (1-cut) * d['lookup'].start_price
    return pd.concat([gross.rename('gross'),
                      months.rename('months_to_sale')], axis=1)


def process_inputs(part, role, delay):
    # load simulated data
    sim = concat_sim_chunks(part,
                            restrict_to_first=False,
                            drop_no_arrivals=True)
    # value components
    sales = get_sales(sim)
    months = get_duration(sim, role)
    idx = months.index
    values = get_discounted_values(sales, months)
    norm_values = values / sim['lookup'].start_price.reindex(
        index=idx, level='lstg')
    # listing features
    x = init_x(part, idx)
    # remove auto accept/reject features from x['lstg'] for buyer models
    if role == BYR_PREFIX:
        x['lstg'].drop(['auto_decline', 'auto_accept', 'has_decline', 'has_accept'],
                       axis=1, inplace=True)
    # thread features
    x_thread = get_x_thread(sim['threads'], idx)
    if delay:
        x_thread[INT_REMAINING] = calculate_remaining(lstg_start=sim['start_time'],
                                                      clock=sim['clock'],
                                                      idx=idx)
    x_thread = add_turn_indicators(x_thread)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)
    # offer features
    x.update(get_x_offer(sim['offers'], idx, role))
    return {'y': norm_values, 'x': x}


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
