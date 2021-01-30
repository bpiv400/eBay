import argparse
import pandas as pd
from agent.eval.util import read_table, save_table
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    get_output_dir, load_valid_data
from utils import safe_reindex
from agent.const import DELTA_BYR
from featnames import X_OFFER, LOOKUP, X_THREAD, START_PRICE, TEST


def get_return(data=None, norm=None):
    """
    Calculates (dollar) discount and sale rate.
    :param dict data: contains DataFrames.
    :param pd.Series norm: normalized sale prices.
    :return: pd.Series of eval stats.
    """
    # first buyer only, sales only
    data = only_byr_agent(data)
    data = safe_reindex(data, idx=norm.index)

    # discount
    sale_norm = get_sale_norm(data[X_OFFER])
    discount = (norm - sale_norm).dropna()

    # dollar discount
    start_price = safe_reindex(data[LOOKUP][START_PRICE],
                               idx=discount.index)
    dollar = discount * start_price

    s = pd.Series()
    s['dollar'] = dollar.mean()
    s['buyrate'] = len(sale_norm) / len(data[X_THREAD])

    return s


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_BYR)
    parser.add_argument('--read', action='store_true')
    params = parser.parse_args()

    run_dir = get_run_dir(byr=True, delta=params.delta)
    if params.read:
        read_table(run_dir=run_dir)
    output = dict()

    # rewards from data
    data = load_valid_data(part=TEST, byr=True)
    norm = get_sale_norm(offers=data[X_OFFER])
    output['Humans'] = get_return(data=data, norm=norm)

    # rewards from heuristic strategy
    heur_dir = get_output_dir(byr=True,
                              delta=params.delta,
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


if __name__ == '__main__':
    main()
