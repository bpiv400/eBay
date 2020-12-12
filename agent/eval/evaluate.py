import argparse
import os
import pandas as pd
from agent.util import get_run_dir, load_values, load_valid_data, \
    get_sale_norm, get_norm_reward
from agent.eval.util import sim_run_dir
from utils import topickle, unpickle, compose_args, safe_reindex
from agent.const import AGENT_PARAMS
from constants import EPS
from featnames import OBS, X_OFFER, TEST, AGENT_PARTITIONS, THREAD, \
    LOOKUP, START_PRICE, INDEX, BYR, DELTA


def get_byr_return(data=None, values=None):
    sale_norm = get_sale_norm(data[X_OFFER].xs(1, level=THREAD,
                                               drop_level=False))
    vals = safe_reindex(values, idx=sale_norm.index)
    norm = vals - sale_norm
    norm = norm.reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = norm * data[LOOKUP][START_PRICE]

    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['value'] = (norm / vals).mean()
    s['sold_pct'] = len(sale_norm.index) / len(data[LOOKUP].index)
    s['sale_norm'] = sale_norm.mean()
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
    args = vars(parser.parse_args())

    run_dir = get_run_dir(**args)
    if args['read']:
        path = run_dir + '{}.pkl'.format(args['part'])
        if os.path.isfile(path):
            df = unpickle(path)
            print(df)
            exit()

    # preliminaries
    values = load_values(part=args['part'], delta=args[DELTA])
    if not args[BYR]:
        values *= args[DELTA]
    f = wrapper(values=values, byr=args[BYR])
    output = dict()

    # rewards from data
    data = load_valid_data(part=args['part'], byr=args[BYR])
    output[OBS] = f(data)

    # rewards from heuristic strategy
    heuristic_dir = sim_run_dir(heuristic=True, **args)
    data = load_valid_data(part=args['part'],
                           run_dir=heuristic_dir,
                           byr=args[BYR])
    if data is not None:
        output['heuristic'] = f(data)

    # rewards from agent run
    data = load_valid_data(part=args['part'], run_dir=run_dir, byr=args[BYR])
    if data is not None:
        output['Agent'] = f(data)

    # save table
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    print(df)
    topickle(df, run_dir + '{}.pkl'.format(args['part']))


if __name__ == '__main__':
    main()
