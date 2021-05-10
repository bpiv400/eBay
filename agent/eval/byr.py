import argparse
import os
from copy import deepcopy
import pandas as pd
from agent.util import get_run_dir, get_sale_norm, only_byr_agent, \
    load_valid_data, get_log_dir, get_sim_dir
from utils import safe_reindex, unpickle, topickle
from agent.const import DELTA_BYR, TURN_COST_CHOICES
from constants import IDX
from featnames import X_OFFER, LOOKUP, X_THREAD, START_PRICE, \
    NORM, CON, REJECT, INDEX, BYR, SLR, TEST


def get_return(data=None, norm=None):
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


def amend_and_process(data=None, norm=None):
    slr_norm = 1 - data[X_OFFER][NORM].unstack()[IDX[SLR]]
    byr_con = data[X_OFFER][CON].unstack()[IDX[BYR]]

    # rejects list price
    data_rej = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=True)
    return_minus = get_return(data=data_rej, norm=norm)

    # accepts list price
    data_acc = amend_outcome(data=data,
                             slr_norm=slr_norm,
                             byr_con=byr_con,
                             reject=False)
    return_plus = get_return(data=data_acc, norm=norm)
    return return_minus, return_plus


def get_keys(delta=None):
    if delta == 1:
        keys = ['$1{}\\epsilon$'.format(sign) for sign in ['-', '+']]
    else:
        keys = ['${}$'.format(delta)]
    return keys


def get_output(d=None, agent_thread=1):
    print('Agent thread {}'.format(agent_thread))
    k = 'thread{}'.format(agent_thread)
    if k not in d:
        d[k] = pd.DataFrame()

    if 'Humans' not in d[k].index:
        data = load_valid_data(byr=True, agent_thread=agent_thread, minimal=True)
        data = only_byr_agent(data=data, agent_thread=agent_thread)
        d[k]['Humans'] = get_return(data=data)

    for delta in DELTA_BYR:
        keys = get_keys(delta=delta)
        if keys[0] in d[k].index:
            continue

        sim_dir = get_run_dir(byr=True, delta=delta)
        if not os.path.isdir(sim_dir):
            continue

        sim_dir = get_sim_dir(byr=True, delta=delta)
        data = load_valid_data(sim_dir=sim_dir,
                               agent_thread=agent_thread,
                               minimal=True)
        data = only_byr_agent(data=data, agent_thread=agent_thread)
        if data is None:
            continue

        if delta == 1:
            keys = ['$1{}\\epsilon$'.format(sign) for sign in ['-', '+']]
            d[k].loc[keys[0], :], d[k].loc[keys[1], :] = \
                amend_and_process(data=data)
        else:
            d[k].loc[keys[0], :] = get_return(data=data)


def get_sales_output(d=None):
    print('Sales only')
    k = 'sales'
    if k not in d:
        d[k] = pd.DataFrame()

    data = load_valid_data(byr=True, minimal=True)
    norm = get_sale_norm(offers=data[X_OFFER])
    if 'Humans' not in d[k].index:
        data = only_byr_agent(data=data)
        data = safe_reindex(data, idx=norm.index)
        d[k]['Humans'] = get_return(data=data, norm=norm)

    for delta in DELTA_BYR:
        keys = get_keys(delta=delta)
        if keys[0] in d[k].index:
            continue

        sim_dir = get_sim_dir(byr=True, delta=delta)
        data = only_byr_agent(load_valid_data(sim_dir=sim_dir, minimal=True))
        if data is None:
            continue
        data = safe_reindex(data, idx=norm.index)

        if delta == 1:
            d[k].loc[keys[0], :], d[k].loc[keys[1], :] = \
                amend_and_process(data=data, norm=norm)
        else:
            d[k].loc[keys[0], :] = get_return(data=data, norm=norm)


def get_heuristic_output(d=None):
    print('Heuristic agents')
    k = 'heuristic'
    if k not in d:
        d[k] = pd.DataFrame()

    for delta in DELTA_BYR:
        keys = get_keys(delta=delta)
        if keys[0] in d[k].index:
            continue

        sim_dir = get_sim_dir(byr=True, heuristic=True, delta=delta)
        if not os.path.isdir(sim_dir):
            continue

        data = only_byr_agent(load_valid_data(sim_dir=sim_dir, minimal=True))
        if data is None:
            continue

        if delta == 1:
            d[k].loc[keys[0], :], d[k].loc[keys[1], :] = \
                amend_and_process(data=data)
        else:
            d[k].loc[keys[0], :] = get_return(data=data)


def get_turn_cost_output(d=None):
    print('Turn cost penalties')
    plus, minus = 'turn_cost_plus', 'turn_cost_minus'
    if plus not in d:
        d[plus], d[minus] = pd.DataFrame(), pd.DataFrame()

    for c in TURN_COST_CHOICES:
        if c == 0:
            continue

        key = '${}'.format(c)
        if key in d[plus].index:
            continue

        sim_dir = get_sim_dir(byr=True, delta=1, turn_cost=c)
        if not os.path.isdir(sim_dir):
            continue

        data = only_byr_agent(load_valid_data(sim_dir=sim_dir, minimal=True))
        if data is None:
            continue

        d[minus].loc[key, :], d[plus].loc[key, :] = \
            amend_and_process(data=data)


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    read = parser.parse_args().read

    # load existing file
    log_dir = get_log_dir(byr=True)
    try:
        d = unpickle(log_dir + '{}.pkl'.format(TEST))
    except FileNotFoundError:
        d = dict()
    if read:
        for k, v in d.items():
            print(k)
            print(v)
        exit()

    # create output statistics
    get_output(d, agent_thread=1)
    get_output(d, agent_thread=2)
    get_sales_output(d)
    get_heuristic_output(d)
    get_turn_cost_output(d)

    # save
    topickle(d, log_dir + '{}.pkl'.format(TEST))


if __name__ == '__main__':
    main()
