import os
import numpy as np
import pandas as pd
from agent.const import FULL, SPARSE, NOCON, DISCOUNT_RATE
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR, VALIDATION, IDX
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, LSTG, CON, \
    LOOKUP, X_THREAD, X_OFFER, START_PRICE
from utils import unpickle, load_file


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(**kwargs):
    # log directory
    log_dir = get_log_dir(kwargs[BYR])

    # run id
    run_id = kwargs['con_set']
    if kwargs['suffix'] is not None:
        run_id += '_{}'.format(kwargs['suffix'])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def define_con_set(con_set=None, byr=False):
    if con_set == FULL:
        num_con = 101
    elif con_set == SPARSE:
        num_con = 11
    elif con_set == NOCON:
        num_con = 2
    else:
        raise ValueError('Invalid concession set: {}'.format(con_set))

    cons = np.arange(num_con) / (num_con - 1)
    if not byr:
        cons = np.concatenate([cons, [1.1]])  # expiration rejection
    return cons


def find_best_run(byr=None, sales=False, verbose=True):
    log_dir = get_log_dir(byr=byr)
    df = unpickle(log_dir + 'runs.pkl')
    s = df.xs(VALIDATION, level='part')[NORM]
    s = s.xs('sales' if sales else 'all', level='listings')
    run_id = s[~s.isna()].astype('float64').idxmax()
    if verbose:
        print('Best run: {}'.format(run_id))
    return log_dir + '{}/'.format(run_id)


def get_valid_slr(data=None, lookup=None):
    # count non-automatic seller offers
    auto = data['offers'][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    s = valid.groupby(LSTG).sum()
    lstgs = s[s > 0].index.intersection(lookup.index, sort=None)  # listings to keep
    for k, v in data.items():
        data[k] = v.reindex(index=lstgs, level=LSTG)
    lookup = lookup.reindex(index=lstgs)
    return data, lookup


def get_value_slr(offers=None, start_price=None):
    norm = get_sale_norm(offers)
    norm = norm.reindex(index=start_price.index, fill_value=0.)
    price = norm * start_price
    return norm.mean(), price.mean()


def get_valid_byr(data=None):
    idx_offers = data[X_THREAD][data[X_THREAD]['byr_agent']].index
    idx_delays = data['delays'].xs(0, level='day').index
    lstgs = idx_offers.droplevel(THREAD).union(idx_delays)
    for k, v in data.items():
        if isinstance(v.index, pd.MultiIndex):
            data[k] = v.reindex(index=lstgs, level=LSTG)
        else:
            data[k] = v.reindex(index=lstgs)
    return data


def get_value_byr(data=None, values=None):
    lstgs = data[LOOKUP].index
    norm = get_sale_norm(data[X_OFFER], drop_thread=False)
    agent_norm = norm[data[X_THREAD]['byr_agent']].droplevel(THREAD)
    norm_value = values.mean(axis=1).loc[lstgs] / data[LOOKUP][START_PRICE]
    agent_value = norm_value.loc[agent_norm.index] - agent_norm
    agent_value = agent_value.reindex(index=lstgs, fill_value=0)
    return agent_value.mean(), (agent_value * data[LOOKUP][START_PRICE]).mean()


def get_sale_norm(offers=None, drop_thread=True):
    sale_norm = offers.loc[offers[CON] == 1, NORM]
    # redefine norm to be fraction of list price
    slr_turn = (sale_norm.index.get_level_values(level=INDEX) % 2) == 0
    sale_norm = slr_turn * (1. - sale_norm) + (1. - slr_turn) * sale_norm
    # restrict index levels
    sale_norm = sale_norm.droplevel(INDEX)
    if drop_thread:
        sale_norm = sale_norm.droplevel(THREAD)
    return sale_norm


def load_values(part=None):
    df = load_file(part, 'sim/values')
    v = df.sale_price * DISCOUNT_RATE ** df.relist_ct
    v = v.droplevel('sale')
    if v.index.duplicated().any():
        return v.groupby(LSTG).mean()
    return v
