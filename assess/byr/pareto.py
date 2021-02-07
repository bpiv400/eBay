import pandas as pd
from agent.util import get_log_dir
from utils import topickle, unpickle
from constants import PLOT_DIR
from featnames import TEST


def main():
    df = unpickle(get_log_dir(byr=True) + '{}.pkl'.format(TEST))
    tuples = [s.split('_') for s in list(df.index)]
    idx = pd.MultiIndex.from_tuples(tuples, names=['gamma', 'turn_cost'])
    df.index = idx

    d = {'pareto_discount': df[['buyrate', 'discount']],
         'pareto_dollar': df[['buyrate', 'dollar']],
         'pareto_sales': df[['buyrate_sales', 'dollar_sales']]}
    for k, v in d.items():
        v.columns = ['x', 'y']

    topickle(d, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
