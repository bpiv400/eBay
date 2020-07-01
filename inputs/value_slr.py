import numpy as np
import pandas as pd
from inputs.util import save_files, construct_x_slr, \
    get_sale_norm, create_index_slr, get_init_data
from inputs.policy_slr import input_parameters
from utils import get_cut, slr_reward, max_slr_reward
from inputs.const import DELTA_ACTION, C_ACTION
from constants import MONTH, TRAIN_RL, VALIDATION, TEST
from featnames import META, START_PRICE, START_TIME, CON


def get_duration(idx, delay, data):
    # calculate number of months from listing start to action
    clock = data['clock']
    if delay:
        t_now = clock.groupby(clock.index.names[:-1]).shift()
        t_now = t_now.dropna().astype(clock.dtype)
        t_now = t_now.loc[idx]
    else:
        t_now = clock.loc[idx]
    lstg_start = data['lookup'][START_TIME]
    t_start = lstg_start.reindex(index=idx, level='lstg')
    months = (t_now - t_start) / MONTH \
        + t_start.index.get_level_values(level='sim')
    return months.rename('months')


def get_sales(data):
    norm = get_sale_norm(data['offers'])
    # number of relistings
    relist_count = norm.index.get_level_values(level='sim')
    norm = norm.reset_index(['sim', 'thread', 'index'], drop=True)
    # time of sale
    sale_time = data['clock'][data['offers'][CON] == 1]
    sale_time = sale_time.reset_index(
        sale_time.index.names[1:], drop=True)
    elapsed = (sale_time - data['lookup'][START_TIME]) / MONTH
    assert (elapsed >= 0).all()
    months = relist_count + elapsed
    # gross bin_price and gross actual price
    cut = data['lookup'][META].apply(get_cut)
    bin_gross = (1-cut) * data['lookup'][START_PRICE]
    gross = norm * bin_gross
    return pd.concat([bin_gross.rename('bin_proceeds'),
                      gross.rename('sale_proceeds'),
                      months.rename('months_to_sale')], axis=1)


def process_inputs(part, delay):
    # data
    data = get_init_data(part, sim=True)

    # master index
    idx = create_index_slr(offers=data['offers'], delay=delay)

    # value components
    sales = get_sales(data)
    months_since_start = get_duration(idx, delay, data)
    df = months_since_start.to_frame().join(sales)

    # discounted values
    values = slr_reward(months_to_sale=df.months_to_sale,
                        months_since_start=df.months,
                        sale_proceeds=df.sale_proceeds)

    max_values = max_slr_reward(months_since_start=df.months,
                                bin_proceeds=df.bin_proceeds)

    # normalize by max values
    norm_values = values / max_values
    assert norm_values.max() <= 1

    # integer values
    y = np.maximum(np.round(norm_values * 100), 0.).astype('uint8')

    # input features dictionary
    x = construct_x_slr(part=part, delay=delay,
                        idx=y.index, data=data)

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
