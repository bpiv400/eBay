import pandas as pd
from agent.util import get_log_dir
from agent.eval.util import load_table
from utils import topickle
from constants import PLOT_DIR


def main():
    df = load_table(run_dir=get_log_dir(byr=True))
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
