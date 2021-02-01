import pandas as pd
from assess.util import get_eval_df
from utils import topickle
from agent.const import DELTA_BYR, TURN_COST_CHOICES
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
        for turn_cost in TURN_COST_CHOICES:
            params = dict(byr=True, delta=delta, turn_cost=turn_cost)
            df = get_eval_df(**params)
            if df is None or 'Agent' not in df.index:
                continue
            df_sales = get_eval_df(suffix='sales', **params)

            k = '{}_{}'.format(delta, turn_cost)
            add_lines(output=d['pareto_all'], df=df, k=k)
            add_lines(output=d['pareto_sales'], df=df_sales, k=k)

    print(d)

    topickle(d, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
