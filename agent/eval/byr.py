import argparse
import os
import pandas as pd
from agent.eval.util import read_table, save_table
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    get_output_dir, load_valid_data
from utils import safe_reindex
from agent.const import DELTA_BYR, TURN_COST_CHOICES
from featnames import X_OFFER, LOOKUP, X_THREAD, START_PRICE, TEST, \
    DELTA, TURN_COST


def calculate_stats(data=None, norm=None):
    # discount
    sale_norm = get_sale_norm(data[X_OFFER])
    if norm is None:
        norm = 1
    discount = (norm - sale_norm).dropna()

    # dollar discount
    start_price = safe_reindex(data[LOOKUP][START_PRICE],
                               idx=discount.index)
    dollar = discount * start_price

    s = pd.Series()
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
    # first buyer only
    data = only_byr_agent(data)

    s = calculate_stats(data=data)

    data = safe_reindex(data, idx=norm.index)
    s = s.append(calculate_stats(data=data, norm=norm).add_suffix('_sales'))

    return s


def create_output(delta=None, turn_cost=None, read=False):
    run_dir = get_run_dir(byr=True, delta=delta, turn_cost=turn_cost)
    if not os.path.isdir(run_dir):
        return

    if read:
        read_table(run_dir=run_dir)
        exit()
    output = dict()

    # rewards from data
    data = load_valid_data(part=TEST, byr=True)
    norm = get_sale_norm(offers=data[X_OFFER])
    output['Humans'] = get_return(data=data, norm=norm)

    # rewards from heuristic strategy
    heur_dir = get_output_dir(byr=True,
                              delta=delta,
                              turn_cost=turn_cost,
                              heuristic=True,
                              part=TEST)
    data = load_valid_data(part=TEST, run_dir=heur_dir)
    if data is not None:
        output['Heuristic'] = get_return(data=data, norm=norm)

    # rewards from agent run
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if data is not None:
        output['Agent'] = get_return(data=data, norm=norm)

    save_table(run_dir=run_dir, output=output)


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_BYR)
    parser.add_argument('--turn_cost', type=int, default=0,
                        choices=TURN_COST_CHOICES)
    parser.add_argument('--read', action='store_true')
    params = vars(parser.parse_args())

    if params[DELTA] is None:
        for delta in DELTA_BYR:
            for turn_cost in TURN_COST_CHOICES:
                print('(delta, turn cost): ({}, {})'.format(
                    delta, turn_cost))
                params[DELTA] = delta
                params[TURN_COST] = turn_cost
                create_output(**params)
    else:
        create_output(**params)


if __name__ == '__main__':
    main()
