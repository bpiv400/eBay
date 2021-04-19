import argparse
import os
from copy import deepcopy
import pandas as pd
from agent.eval.util import load_table, print_table, save_table
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    load_valid_data, get_log_dir, get_output_dir
from utils import safe_reindex
from agent.const import DELTA_BYR, TURN_COST_CHOICES
from constants import IDX
from featnames import X_OFFER, LOOKUP, X_THREAD, START_PRICE, \
    NORM, CON, REJECT, INDEX, BYR, SLR


def calculate_stats(data=None, norm=None):
    # discount
    sale_norm = get_sale_norm(data[X_OFFER])
    if norm is None:
        norm = 1
    else:
        norm = safe_reindex(norm, idx=sale_norm.index)
        assert norm.isna().sum() == 0
    discount = norm - sale_norm

    # dollar discount
    start_price = safe_reindex(data[LOOKUP][START_PRICE],
                               idx=discount.index)
    dollar = discount * start_price

    s = pd.Series()
    s['discount'] = discount.mean()
    s['dollar'] = dollar.mean()
    s['buyrate'] = len(sale_norm) / len(data[X_THREAD])

    return s


def get_return(data=None, norm=None):
    """
    Calculates (dollar) discount and sale rate.
    :param dict data: contains DataFrames.
    :param pd.Series norm: normalized sale prices.
    :return: pd.Series of eval stats.
    """
    s = calculate_stats(data=data)

    data = safe_reindex(data, idx=norm.index)
    s = s.append(calculate_stats(data=data, norm=norm).add_suffix('_sales'))

    return s


def amend_outcome(data=None, byr_con=None, slr_norm=None, reject=True):
    data_alt = deepcopy(data)
    for t in IDX[BYR]:
        mask = byr_con[t] == int(reject)
        if t > 1:
            mask = mask & (slr_norm[t-1] == 1)
        mask = mask.to_frame().assign(index=t).set_index(
            INDEX, append=True).squeeze()
        idx = mask[mask].index
        data_alt[X_OFFER].loc[idx, [CON, NORM, REJECT]] = \
            [float(~reject), float(~reject), reject]
    return data_alt


def amend_and_process(data=None, norm=None):
    slr_norm = 1 - data[X_OFFER][NORM].unstack()[IDX[SLR]]
    byr_con = data[X_OFFER][CON].unstack()[IDX[BYR]]

    # rejects list price
    data_rej = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=True)
    return_minus = get_return(data=data_rej, norm=norm)

    # accepts list price
    data_acc = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=False)
    return_plus = get_return(data=data_acc, norm=norm)
    return return_minus, return_plus


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    read = parser.parse_args().read
    overwrite = parser.parse_args().overwrite

    log_dir = get_log_dir(byr=True)
    df = load_table(run_dir=log_dir)
    if read:
        if df is None:
            print('No output table exists.')
        else:
            print_table(df, byr=True)
        exit()

    if overwrite or df is None:
        output = dict()
    else:
        output = df.to_dict(orient=INDEX)

    # rewards from data
    data = load_valid_data(byr=True, minimal=True)
    norm = get_sale_norm(offers=data[X_OFFER])
    if 'Humans' not in output:
        data = only_byr_agent(data)
        output['Humans'] = get_return(data=data, norm=norm)

    for delta in DELTA_BYR:
        for turn_cost in TURN_COST_CHOICES:
            run_dir = get_run_dir(byr=True,
                                  delta=delta,
                                  turn_cost=turn_cost)
            if not os.path.isdir(run_dir):
                continue

            print('(delta, turn cost): ({}, {})'.format(
                delta, turn_cost))
            data = only_byr_agent(load_valid_data(run_dir=run_dir,
                                                  minimal=True))

            h_dir = get_output_dir(byr=True, heuristic=True,
                                   delta=delta, turn_cost=turn_cost)
            data_h = only_byr_agent(load_valid_data(run_dir=h_dir,
                                                    minimal=True))

            if data is not None:
                if delta == 1:
                    keys = ['$1{}\\epsilon$_${}'.format(sign, turn_cost)
                            for sign in ['-', '+']]
                    if keys[0] not in output:
                        output[keys[0]], output[keys[1]] = \
                            amend_and_process(data=data, norm=norm)

                    keys_h = ['h: {}'.format(k) for k in keys]
                    if data_h is not None and keys_h[0] not in output:
                        output[keys_h[0]], output[keys_h[1]] = \
                            amend_and_process(data=data_h, norm=norm)

                else:
                    key = '${}$_${}'.format(delta, turn_cost)
                    if key not in output:
                        output[key] = get_return(data=data, norm=norm)

                    key_h = 'h: {}'.format(key)
                    if data_h is not None and key_h not in output:
                        output[key_h] = get_return(
                            data=data_h, norm=norm)

    save_table(run_dir=log_dir, output=output)


if __name__ == '__main__':
    main()
