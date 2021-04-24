import pandas as pd
from agent.util import get_log_dir
from utils import unpickle, topickle
from constants import PLOT_DIR
from featnames import TEST


def main():
    d = unpickle(get_log_dir(byr=True) + '{}.pkl'.format(TEST))

    output = {'discount': d['thread1'][['buyrate', 'discount']],
              'dollar': d['thread1'][['buyrate', 'dollar']],
              'sales': d['sales'][['buyrate', 'dollar']]}

    for name in ['minus', 'plus']:
        key = 'cost_{}'.format(name)
        output[key] = pd.concat([d['thread1'], d['turn_{}'.format(key)]])
        output[key] = output[key][['buyrate', 'discount']]

    for k, v in output.items():
        v.columns = ['x', 'y']

    keys = list(output.keys())
    for k in keys:
        output['pareto_{}'.format(k)] = output.pop(k)

    topickle(output, PLOT_DIR + 'byrpareto.pkl')


if __name__ == '__main__':
    main()
