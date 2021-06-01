import numpy as np
from agent.eval.util import get_eval_path
from assess.util import save_dict
from utils import unpickle
from agent.const import BYR_CONS


def get_pareto(df0=None):
    df = df0.sort_values('buyrate', ascending=False)
    df['max_discount'] = df['discount'].cummax()
    df['max_dollar'] = df['dollar'].cummax()
    df = df.sort_values('discount', ascending=False)
    df['max_buyrate1'] = df['buyrate'].cummax()
    df = df.sort_values('dollar', ascending=False)
    df['max_buyrate2'] = df['buyrate'].cummax()
    pareto = ((df['max_buyrate1'] == df['buyrate']) &
              (df['max_buyrate2'] == df['buyrate']) &
              (df['max_discount'] == df['discount']) &
              (df['max_dollar'] == df['dollar']))
    return df.loc[pareto, ['discount', 'dollar', 'buyrate']]


def main():
    d = unpickle(get_eval_path(byr=True))

    # restrict heuristics
    df = get_pareto(d['heuristic'])
    heuristics = []
    for idx in d['full'].index:
        if idx != 'Humans':
            s = d['full'].loc[idx]
            dist = (df['discount'] - s['discount']) ** 2 \
                + (df['buyrate'] - s['buyrate']) ** 2
            heuristics.append(dist.index[np.argmin(dist)])
    d['heuristic'] = d['heuristic'].loc[heuristics]

    # print heuristics
    for heur in d['heuristic'].index:
        print('{}: {}'.format(heur, BYR_CONS.loc[int(heur[:-1])].to_dict()))

    # discount in percentage terms
    for k in d.keys():
        d[k].loc[:, 'discount'] *= 100

    output = {}
    for name in ['discount', 'dollar']:
        key = 'pareto_{}'.format(name)
        output[key] = {}
        for k in ['full', 'heuristic']:
            output[key][k] = d[k].loc[:, ['buyrate', name]]

    # best observed offer
    output['pareto_best'] = {}
    for k in ['full', 'heuristic']:
        mask = d[k]['dollar_best'] > -2
        output['pareto_best'][k] = d[k].loc[mask, ['buyrate_best', 'dollar_best']]

    # turn cost agents
    key = 'pareto_cost'
    output[key] = {}
    output[key]['full'] = d['full'].loc[:, ['buyrate', 'discount']]
    output[key]['plus'] = d['turn_cost_plus'].loc[:, ['buyrate', 'discount']]
    output[key]['plus'].loc['$0', :] = output[key]['full'].loc['$1+\\epsilon$', :]
    output[key]['plus'].sort_index(inplace=True)

    # rename columns
    for key in output.keys():
        for k in output[key].keys():
            output[key][k].columns = ['x', 'y']

    # save
    save_dict(output, 'byrpareto')


if __name__ == '__main__':
    main()
