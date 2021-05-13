import pandas as pd
from agent.eval.util import get_eval_path
from utils import unpickle, topickle
from constants import PLOT_DIR


def main():
    d = unpickle(get_eval_path(byr=True))

    output = {'discount': d['full'][['buyrate', 'discount']],
              'dollar': d['full'][['buyrate', 'dollar']],
              'sales': d['sales'][['buyrate', 'dollar']]}

    for name in ['minus', 'plus']:
        key = 'turn_cost_{}'.format(name)
        if len(d[key]) > 0:
            output[key] = pd.concat([d['full'], d[key]])
            output[key] = output[key][['buyrate', 'discount']]

    for k, v in output.items():
        v.columns = ['x', 'y']

    keys = list(output.keys())
    for k in keys:
        output['pareto_{}'.format(k)] = output.pop(k)

    topickle(output, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
