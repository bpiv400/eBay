import argparse
import os
import pandas as pd
from agent.util import get_log_dir, load_values, load_valid_data, \
    get_sale_norm, get_norm_reward
from utils import topickle, unpickle, compose_args, safe_reindex
from agent.const import AGENT_PARAMS
from constants import EPS
from featnames import OBS, X_OFFER, TEST, AGENT_PARTITIONS, THREAD, \
    LOOKUP, START_PRICE, BYR_AGENT, X_THREAD, INDEX


def get_byr_return(data=None, values=None):
    sale_norm = get_sale_norm(data[X_OFFER], drop_thread=False)
    if BYR_AGENT in data[X_THREAD].columns:
        sale_norm = sale_norm[data[X_THREAD][BYR_AGENT]].droplevel(THREAD)
    else:
        sale_norm = sale_norm.xs(1, level=THREAD)
    norm = safe_reindex(values, idx=sale_norm.index) - sale_norm
    norm = norm.reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = norm * data[LOOKUP][START_PRICE]

    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['sold_pct'] = len(sale_norm.index) / len(data[LOOKUP].index)
    return s


def get_slr_return(data=None, values=None):
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
    s['sold_pct'] = len(sale_norm) / (len(sale_norm) + len(cont_value))
    s['norm_cont'] = cont_value.mean()
    s['dollar_cont'] = dollar.loc[cont_value.index].mean()
    s['norm_sold'] = sale_norm.mean()
    s['dollar_sold'] = dollar.loc[sale_norm.index].mean()
    s['net_norm'] = net_norm.mean()
    s['dollar_norm'] = dollar_norm.mean()
    return s


def wrapper(values=None, byr=False):
    return_func = get_byr_return if byr else get_slr_return
    return lambda d: return_func(data=d, values=values)


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--read', action='store_true')
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    args = parser.parse_args()

    log_dir = get_log_dir(byr=args.byr, delta=args.delta)
    if args.read:
        path = log_dir + '{}.pkl'.format(args.part)
        if os.path.isfile(path):
            df = unpickle(path)
            print(df)
            exit()

    # preliminaries
    values = load_values(part=args.part, delta=args.delta)
    if not args.byr:
        values *= args.delta
    f = wrapper(values=values, byr=args.byr)
    output = dict()

    # rewards from data
    data = load_valid_data(part=args.part, byr=args.byr)
    output[OBS] = f(data)

    # rewards to first non-agent buyer
    if args.byr:
        pass

    # rewards from heuristic strategy
    heuristic_dir = get_log_dir(**vars(args)) + 'heuristic/'
    data = load_valid_data(part=args.part, run_dir=heuristic_dir, byr=args.byr)
    if data is not None:
        output['heuristic'] = f(data)

    # rewards from agent runs
    run_ids = [p for p in os.listdir(log_dir)
               if os.path.isdir(log_dir + p)]
    for run_id in run_ids:
        run_dir = log_dir + '{}/'.format(run_id)
        data = load_valid_data(part=args.part, run_dir=run_dir, byr=args.byr)
        if data is not None:
            output[run_id] = f(data)

    # save table
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    print(df)
    topickle(df, log_dir + '{}.pkl'.format(args.part))


if __name__ == '__main__':
    main()
