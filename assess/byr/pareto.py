import pandas as pd
from agent.eval.util import get_eval_path
from utils import unpickle, topickle
from constants import PLOT_DIR


def main():
    d = unpickle(get_eval_path(byr=True))

    df = d['heuristic'].sort_values('buyrate', ascending=False)
    df['max_discount'] = df['discount'].cummax()
    df = df.sort_values('discount', ascending=False)
    df['max_buyrate'] = df['buyrate'].cummax()
    pareto = (df['max_buyrate'] == df['buyrate']) & \
             (df['max_discount'] == df['discount'])
    d['heuristic'] = df.loc[pareto, ['discount', 'dollar', 'buyrate']]

    output = {}
    for name in ['discount', 'dollar', 'sales']:
        key = 'pareto_{}'.format(name)
        output[key] = {}
        for k in ['full', 'heuristic']:
            if k == 'heuristic' and name == 'sales':
                continue
            x = 'buyrate_sales' if name == 'sales' else 'buyrate'
            y = 'dollar_sales' if name == 'sales' else name
            output[key][k] = d[k].loc[:, [x, y]]
            output[key][k].columns = ['x', 'y']
            if name == 'discount':
                output[key][k]['y'] *= 100

    # for name in ['minus', 'plus']:
    #     key = 'turn_cost_{}'.format(name)
    #     if len(d[key]) > 0:
    #         output[key] = pd.concat([d['full'], d[key]])
    #         output[key] = output[key][['buyrate', 'discount']]

    topickle(output, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
