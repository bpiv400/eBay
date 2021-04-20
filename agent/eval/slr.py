import argparse
import pandas as pd
from agent.eval.util import save_table
from agent.util import get_run_dir, load_values, get_norm_reward, \
    get_output_dir, load_valid_data
from utils import unpickle
from agent.const import DELTA_SLR
from constants import EPS
from featnames import TEST, LOOKUP, START_PRICE


def get_return(data=None, values=None):
    assert values.max() <= 1 + EPS
    sale_norm, cont_value = get_norm_reward(data=data, values=values)
    norm = pd.concat([sale_norm, cont_value]).sort_index()
    start_price = data[LOOKUP][START_PRICE]
    dollar = norm * start_price
    net_norm = norm - values
    dollar_norm = net_norm * start_price

    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['norm_sale'] = sale_norm.mean()
    s['dollar_sale'] = dollar.loc[sale_norm.index].mean()
    s['sale_pct'] = len(sale_norm) / (len(sale_norm) + len(cont_value))
    s['net_norm'] = net_norm.mean()
    s['dollar_norm'] = dollar_norm.mean()
    return s


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--delta', type=float,
                        choices=DELTA_SLR, required=True)
    params = parser.parse_args()
    delta = params.delta

    run_dir = get_run_dir(delta=delta)
    if params.read:
        path = run_dir + '{}.pkl'.format(TEST)
        try:
            df = unpickle(path)
            print(df)
        except FileNotFoundError:
            print('{} not found.'.format(path))

    # preliminaries
    values = delta * load_values(delta=delta)
    output = dict()

    # rewards from data
    data = load_valid_data(byr=False)
    output['Humans'] = get_return(data=data, values=values)

    # rewards from heuristic strategy
    heuristic_dir = get_output_dir(heuristic=True, delta=delta)
    data = load_valid_data(part=TEST, run_dir=heuristic_dir)
    if data is not None:
        output['Heuristic'] = get_return(data=data, values=values)

    # rewards from agent run
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if data is not None:
        output['Agent'] = get_return(data=data, values=values)

    save_table(run_dir=run_dir, output=output)


if __name__ == '__main__':
    main()
