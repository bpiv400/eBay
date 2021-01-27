import argparse
import pandas as pd
from agent.eval.util import read_table, save_table
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    get_output_dir, load_valid_data
from utils import safe_reindex
from featnames import X_OFFER, LOOKUP, START_PRICE, TEST, THREAD


def create_output(data=None, suffix=None):
    sale_norm = get_sale_norm(data[X_OFFER])
    discount = (1 - sale_norm).reindex(
        index=data[LOOKUP].index, fill_value=0)
    dollar = discount * data[LOOKUP][START_PRICE]
    s = pd.Series()
    s['discount'] = discount.mean()
    s['dollar'] = dollar.mean()
    s['buyrate'] = len(sale_norm) / len(data[LOOKUP])
    if suffix is not None:
        s = s.add_suffix('_{}'.format(suffix))
    return s


def get_return(data=None, idx=None):
    # first buyer only
    data = only_byr_agent(data)

    # all listings
    s = create_output(data=data)

    # only sales
    idx0 = idx.droplevel(THREAD)
    data = safe_reindex(data, idx=idx0)
    s = s.append(create_output(data=data, suffix='sales'))

    # only sales to first buyer
    idx1 = pd.Series(index=idx).xs(1, level=THREAD).index
    data = safe_reindex(data, idx=idx1)
    s = s.append(create_output(data=data, suffix='sales1'))

    return s


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    params = parser.parse_args()

    run_dir = get_run_dir(byr=True, delta=1.)
    if params.read:
        read_table(run_dir=run_dir)
    output = dict()

    # rewards from data
    data = load_valid_data(part=TEST, byr=True)
    idx = get_sale_norm(offers=data[X_OFFER], drop_thread=False).index
    output['Humans'] = get_return(data=data, idx=idx)

    # rewards from heuristic strategy
    heur_dir = get_output_dir(byr=True, delta=1., heuristic=True, part=TEST)
    data = load_valid_data(part=TEST, run_dir=heur_dir)
    if data is not None:
        output['Heuristic'] = get_return(data=data, idx=idx)

    # rewards from agent run
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if data is not None:
        output['Agent'] = get_return(data=data, idx=idx)

    save_table(run_dir=run_dir, output=output)


if __name__ == '__main__':
    main()
