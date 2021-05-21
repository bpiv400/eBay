import os
from copy import deepcopy
import pandas as pd
from agent.util import get_sale_norm, load_valid_data, get_sim_dir
from agent.eval.util import eval_args, get_eval_path
from utils import safe_reindex, unpickle, topickle, load_data
from agent.const import DELTA_BYR, TURN_COST_CHOICES
from constants import IDX
from featnames import X_OFFER, LOOKUP, X_THREAD, START_PRICE, NORM, CON, REJECT, \
    INDEX, BYR, SLR

COLS = ['discount', 'dollar', 'buyrate',
        'discount_sales', 'dollar_sales', 'buyrate_sales']


def calculate_return(data=None, norm=None):
    # discount
    sale_norm = get_sale_norm(data[X_OFFER])
    if norm is None:
        norm = 1
    else:
        norm = safe_reindex(norm, idx=sale_norm.index)
        assert norm.isna().sum() == 0
    discount = norm - sale_norm
    # dollar discount
    start_price = safe_reindex(data[LOOKUP][START_PRICE],
                               idx=discount.index)
    dollar = discount * start_price
    s = pd.Series()
    s['discount'] = discount.mean()
    s['dollar'] = dollar.mean()
    s['buyrate'] = len(sale_norm) / len(data[X_THREAD])
    return s


def get_return(data=None, norm=None):
    s0 = calculate_return(data=data)
    data = safe_reindex(data, idx=norm.index, dropna=True)
    s1 = calculate_return(data=data, norm=deepcopy(norm)).rename(
        lambda x: '{}_sales'.format(x), axis=1)
    s = pd.concat([s0, s1])
    return s


def wrapper():
    norm = get_sale_norm(offers=load_data()[X_OFFER])
    return lambda d: get_return(data=d, norm=norm)


def amend_outcome(data=None, byr_con=None, slr_norm=None, reject=True):
    data_alt = deepcopy(data)
    for t in IDX[BYR]:
        mask = byr_con[t] == int(reject)
        if t > 1:
            mask = mask & (slr_norm[t-1] == 1)
        mask = mask.to_frame().assign(index=t).set_index(
            INDEX, append=True).squeeze()
        idx = mask[mask].index
        data_alt[X_OFFER].loc[idx, [CON, NORM, REJECT]] = \
            [float(~reject), float(~reject), reject]
    return data_alt


def amend_and_process(data=None, f=None):
    slr_norm = 1 - data[X_OFFER][NORM].unstack()[IDX[SLR]]
    byr_con = data[X_OFFER][CON].unstack()[IDX[BYR]]

    # rejects list price
    data_rej = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=True)
    return_minus = f(data_rej)

    # accepts list price
    data_acc = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=False)
    return_plus = f(data_acc)
    return return_minus, return_plus


def get_keys(delta=None):
    if delta == 1:
        keys = ['$1{}\\epsilon$'.format(sign) for sign in ['-', '+']]
    else:
        keys = ['${}$'.format(delta)]
    return keys


def get_output(d=None, f=None):
    print('Full agent')
    k = 'full'
    if k not in d:
        d[k] = pd.DataFrame(columns=COLS)

    if 'Humans' not in d[k].index:
        data = load_valid_data(byr=True, minimal=True)
        d[k].loc['Humans', :] = f(data)

    for delta in DELTA_BYR:
        keys = get_keys(delta=delta)
        if keys[0] in d[k].index:
            continue

        sim_dir = get_sim_dir(byr=True, delta=delta)
        data = load_valid_data(sim_dir=sim_dir, minimal=True)
        if data is None:
            continue

        if delta == 1:
            keys = ['$1{}\\epsilon$'.format(sign) for sign in ['-', '+']]
            d[k].loc[keys[0], :], d[k].loc[keys[1], :] = \
                amend_and_process(data=data, f=f)
        else:
            d[k].loc[keys[0], :] = f(data)


def get_heuristic_output(d=None, f=None):
    print('Heuristic agents')
    k = 'heuristic'
    if k not in d:
        d[k] = pd.DataFrame(columns=COLS)

    for delta in DELTA_BYR:
        keys = get_keys(delta=delta)
        if keys[0] in d[k].index:
            continue

        sim_dir = get_sim_dir(byr=True, heuristic=True, delta=delta)
        if not os.path.isdir(sim_dir):
            continue

        data = load_valid_data(sim_dir=sim_dir, minimal=True)
        if data is None:
            continue

        if delta == 1:
            d[k].loc[keys[0], :], d[k].loc[keys[1], :] = \
                amend_and_process(data=data, f=f)
        else:
            d[k].loc[keys[0], :] = f(data)


def get_turn_cost_output(d=None, f=None):
    print('Turn cost penalties')
    plus, minus = 'turn_cost_plus', 'turn_cost_minus'
    if plus not in d:
        d[plus], d[minus] = pd.DataFrame(columns=COLS), pd.DataFrame(columns=COLS)

    for c in TURN_COST_CHOICES:
        if c == 0:
            continue

        key = '${}'.format(c)
        if key in d[plus].index:
            continue

        sim_dir = get_sim_dir(byr=True, delta=1, turn_cost=c)
        if not os.path.isdir(sim_dir):
            continue

        data = load_valid_data(sim_dir=sim_dir, minimal=True)
        if data is None:
            continue

        d[minus].loc[key, :], d[plus].loc[key, :] = \
            amend_and_process(data=data, f=f)


def main():
    args = eval_args()

    # load existing file
    path = get_eval_path(byr=True)
    try:
        d = unpickle(path)
    except FileNotFoundError:
        d = dict()
    if args.read:
        for k, v in d.items():
            print(k)
            print(v)
        exit()

    # wrapper function with normalized sale price in the data
    f = wrapper()

    # create output statistics
    get_output(d, f)
    get_heuristic_output(d, f)
    get_turn_cost_output(d, f)

    # save
    topickle(d, path)


if __name__ == '__main__':
    main()
