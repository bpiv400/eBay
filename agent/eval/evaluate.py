import argparse
import os
import pandas as pd
from agent.util import get_run_dir, load_values, load_valid_data, \
    get_sale_norm, get_norm_reward, only_byr_agent
from agent.eval.util import get_output_dir
from utils import topickle, unpickle, compose_args, safe_reindex
from agent.const import AGENT_PARAMS
from constants import EPS
from featnames import X_OFFER, TEST, AGENT_PARTITIONS, CON, LSTG, \
    LOOKUP, START_PRICE, INDEX, BYR, DELTA, TURN_COST


def get_byr_return(data=None, values=None, turn_cost=None):
    data = only_byr_agent(data=data)
    df = data[X_OFFER]

    # without turn cost penalty
    sale_norm = get_sale_norm(df)
    vals = safe_reindex(values, idx=sale_norm.index)
    norm = vals - sale_norm
    norm = norm.reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = norm * data[LOOKUP][START_PRICE]

    # add in turn cost penalty
    if turn_cost > 0:
        con = df.loc[df.index.isin([1, 3, 5], level=INDEX), CON]
        is_con = (con > 0) & (con < 1)
        penalty = (is_con.groupby(LSTG).sum() * turn_cost).reindex(
            index=data[LOOKUP].index, fill_value=0)
        dollar -= penalty
        norm -= penalty / data[LOOKUP][START_PRICE]

    # output
    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['value'] = (norm / vals).mean()
    s['sold_pct'] = len(sale_norm.index) / len(data[LOOKUP].index)
    s['sale_norm'] = sale_norm.mean()
    return s


def get_slr_return(data=None, values=None, turn_cost=None):
    # TODO: implement turn cost for seller
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


def wrapper(values=None, **params):
    return_func = get_byr_return if params[BYR] else get_slr_return
    return lambda d: return_func(data=d,
                                 values=values,
                                 turn_cost=params[TURN_COST])


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--read', action='store_true')
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    params = vars(parser.parse_args())

    run_dir = get_run_dir(**params)
    if params['read']:
        path = run_dir + '{}.pkl'.format(params['part'])
        if os.path.isfile(path):
            df = unpickle(path)
            print(df)
            exit()

    # preliminaries
    values = load_values(part=params['part'], delta=params[DELTA])
    if not params[BYR]:
        values *= params[DELTA]
    f = wrapper(values=values, **params)
    output = dict()

    # rewards from data
    data = load_valid_data(part=params['part'], byr=params[BYR])
    output['Humans'] = f(data)

    # rewards from heuristic strategy
    heuristic_dir = get_output_dir(heuristic=True, **params)
    data = load_valid_data(part=params['part'],
                           run_dir=heuristic_dir,
                           byr=params[BYR])
    if data is not None:
        output['Heuristic'] = f(data)

    # rewards from agent run
    data = load_valid_data(part=params['part'], run_dir=run_dir, byr=params[BYR])
    if data is not None:
        output['Agent'] = f(data)

    # save table
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    print(df)
    topickle(df, run_dir + '{}.pkl'.format(params['part']))


if __name__ == '__main__':
    main()
