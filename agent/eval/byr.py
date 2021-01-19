import argparse
import pandas as pd
from agent.eval.util import read_table, collect_output
from agent.util import get_run_dir, get_sale_norm, only_byr_agent
from featnames import X_OFFER, LOOKUP, START_PRICE


def get_return(data=None):
    data = only_byr_agent(data=data)
    sale_norm = get_sale_norm(data[X_OFFER])
    discount = (1-sale_norm).reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = discount * data[LOOKUP][START_PRICE]
    # output
    s = pd.Series()
    s['norm'] = discount.mean()
    s['dollar'] = dollar.mean()
    s['norm_sale'] = discount.loc[sale_norm.index].mean()
    s['dollar_sale'] = dollar.loc[sale_norm.index].mean()
    s['sale_pct'] = len(sale_norm.index) / len(data[LOOKUP].index)
    return s


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    params = parser.parse_args()

    run_dir = get_run_dir()
    if params.read:
        read_table(run_dir=run_dir)

    # create and save table
    collect_output(run_dir=run_dir, f=get_return)


if __name__ == '__main__':
    main()
