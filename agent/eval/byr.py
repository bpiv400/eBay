import argparse
import pandas as pd
from agent.eval.util import read_table, collect_output
from agent.util import get_run_dir, get_sale_norm, only_byr_agent
from utils import safe_reindex
from featnames import X_OFFER, CON, THREAD, LOOKUP, START_PRICE, INDEX


def get_return(data=None):
    data = only_byr_agent(data=data)
    df = data[X_OFFER].droplevel(THREAD)

    sale_norm = get_sale_norm(df)
    vals = safe_reindex(data[LOOKUP][START_PRICE], idx=sale_norm.index)
    norm = vals - sale_norm
    norm = norm.reindex(index=data[LOOKUP].index, fill_value=0)
    dollar = norm * data[LOOKUP][START_PRICE]
    norm_value = norm / vals
    walk = df[CON].xs(1, level=INDEX) == 0

    # output
    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['value'] = norm_value.mean()
    s['norm_offer'] = norm[~walk].mean()
    s['dollar_offer'] = dollar[~walk].mean()
    s['value_offer'] = norm_value[~walk].mean()
    s['offer_pct'] = 1 - walk.mean()
    s['norm_sale'] = norm.loc[sale_norm.index].mean()
    s['dollar_sale'] = dollar.loc[sale_norm.index].mean()
    s['value_sale'] = norm_value.loc[sale_norm.index].mean()
    s['sale_pct'] = len(sale_norm.index) / len(data[LOOKUP].index)
    s['sale_norm'] = sale_norm.mean()
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
