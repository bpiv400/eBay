import argparse
import pandas as pd
from agent.eval.util import read_table, save_table
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    get_output_dir, load_valid_data
from utils import safe_reindex
from featnames import X_OFFER, LOOKUP, START_PRICE, TEST


def create_output(data=None, suffix=None):
    sale_norm = get_sale_norm(data[X_OFFER])
    discount = (1 - sale_norm).reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = discount * data[LOOKUP][START_PRICE]
    s = pd.Series()
    s['norm'] = discount.mean()
    s['dollar'] = dollar.mean()
    s['buy_rate'] = len(sale_norm) / len(data[LOOKUP])
    if suffix is not None:
        s = s.add_suffix('_{}'.format(suffix))
    return s


def get_return(data=None, lstgs=None):
    # restrict to first thread
    data = only_byr_agent(data=data)

    # all valid listings
    s = create_output(data=data)

    # only sales in data
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=lstgs)
    s = s.append(create_output(data=data, suffix='sales'))

    return s


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    params = parser.parse_args()

    run_dir = get_run_dir()
    if params.read:
        read_table(run_dir=run_dir)
    output = dict()

    # rewards from data
    data = load_valid_data(part=TEST, byr=True)
    lstgs = get_sale_norm(data[X_OFFER]).index
    output['Humans'] = get_return(data=data, lstgs=lstgs)

    # rewards from heuristic strategy
    heuristic_dir = get_output_dir(heuristic=True, part=TEST)
    data = load_valid_data(part=TEST, run_dir=heuristic_dir)
    if data is not None:
        output['Heuristic'] = get_return(data=data, lstgs=lstgs)

    # rewards from agent run
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if data is not None:
        output['Agent'] = get_return(data=data, lstgs=lstgs)

    save_table(run_dir=run_dir, output=output)


if __name__ == '__main__':
    main()
