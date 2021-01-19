import argparse
import pandas as pd
from agent.eval.util import read_table, collect_output
from agent.util import get_run_dir, load_values, get_norm_reward
from agent.const import DELTA_CHOICES
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
                        choices=DELTA_CHOICES, required=True)
    params = parser.parse_args()
    delta = params.delta

    run_dir = get_run_dir(delta=delta)
    if params.read:
        read_table(run_dir=run_dir)

    # seller values
    values = delta * load_values(part=TEST, delta=delta)

    # create and save table
    collect_output(run_dir=run_dir, delta=delta,
                   f=lambda d: get_return(data=d, values=values))


if __name__ == '__main__':
    main()
