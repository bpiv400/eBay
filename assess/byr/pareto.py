import pandas as pd
from assess.util import get_eval_df
from utils import topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR


def add_lines(output=None, df=None, k=None):
    if len(output) == 0:
        output.loc['Humans', 'x'] = df.loc['Humans', 'buyrate']
        output.loc['Humans', 'y'] = df.loc['Humans', 'dollar']

    k = '{}'.format(k)
    output.loc[k, 'x'] = df.loc['Agent', 'buyrate']
    output.loc[k, 'y'] = df.loc['Agent', 'dollar']


def main():
    d = {'pareto_all': pd.DataFrame(columns=['x', 'y']),
         'pareto_sales': pd.DataFrame(columns=['x', 'y'])}

    for delta in DELTA_BYR:
        df = get_eval_df(byr=True, delta=delta)
        if df is None or 'Agent' not in df.index:
            continue
        df_sales = get_eval_df(byr=True, delta=delta, suffix='sales')

        add_lines(output=d['pareto_all'], df=df, k=delta)
        add_lines(output=d['pareto_sales'], df=df_sales, k=delta)

    topickle(d, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
