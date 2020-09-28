import os
import numpy as np
import pandas as pd
import torch
from agent.const import FULL, SPARSE, NOCON
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR, VALIDATION, IDX
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, LSTG, CON, \
    LOOKUP, X_THREAD, X_OFFER, START_PRICE
from utils import unpickle, load_file, restrict_to_lstgs


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None, delta=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/delta_{}/'.format(role, delta)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(**kwargs):
    # log directory
    log_dir = get_log_dir(byr=kwargs[BYR], delta=kwargs['delta'])

    # run id
    run_id = '{}_{}'.format(kwargs['con_set'], kwargs['entropy'])
    if kwargs['suffix'] is not None:
        run_id += '_{}'.format(kwargs['suffix'])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def define_con_space(con_set=None, byr=False):
    if con_set == FULL:
        num_con = 101
    elif con_set == SPARSE:
        num_con = 11
    elif con_set == NOCON:
        num_con = 2
    else:
        raise ValueError('Invalid concession set: {}'.format(con_set))

    con_space = np.arange(num_con) / (num_con - 1)
    if not byr:
        con_space = np.concatenate([con_space, [1.1]])  # expiration rejection
    return con_space


def find_best_run(byr=None, sales=False, verbose=True):
    log_dir = get_log_dir(byr=byr)
    df = unpickle(log_dir + 'runs.pkl')
    s = df.xs(VALIDATION, level='part')[NORM]
    s = s.xs('sales' if sales else 'all', level='listings')
    run_id = s[~s.isna()].astype('float64').idxmax()
    if verbose:
        print('Best run: {}'.format(run_id))
    return log_dir + '{}/'.format(run_id)


def get_slr_valid(data=None):
    # count non-automatic seller offers
    auto = data[X_OFFER][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    s = valid.groupby(LSTG).sum()
    lstgs = s[s > 0].index.intersection(data[LOOKUP].index, sort=None)
    for k, v in data.items():
        data[k] = restrict_to_lstgs(obj=v, lstgs=lstgs)
    return data


def get_slr_reward(data=None, values=None, delta=None):
    sale_norm = get_sale_norm(data[X_OFFER])
    start_price = data[LOOKUP][START_PRICE]
    sale_price = sale_norm * start_price.reindex(index=sale_norm.index)
    cont_value = (delta * values).reindex(index=start_price.index)
    no_sale = cont_value.drop(sale_price.index)
    reward = pd.concat([sale_price, no_sale]).sort_index()
    assert not reward.index.duplicated().max()
    norm_reward = reward / start_price
    return norm_reward.mean(), reward.mean()


def get_byr_valid(data=None):
    idx_offers = data[X_THREAD][data[X_THREAD]['byr_agent']].index
    idx_delays = data['delays'].xs(0, level='day').index
    lstgs = idx_offers.droplevel(THREAD).union(idx_delays)
    for k, v in data.items():
        data[k] = restrict_to_lstgs(obj=v, lstgs=lstgs)
    return data


def get_byr_reward(data=None, values=None):
    start_price = data[LOOKUP][START_PRICE]
    lstgs = start_price.index
    norm_value = values.mean(axis=1).loc[lstgs] / start_price
    sale_norm = get_sale_norm(data[X_OFFER], drop_thread=False)
    if X_THREAD in data and 'byr_agent' in data[X_THREAD].columns:
        sale_norm = sale_norm[data[X_THREAD]['byr_agent']]
    sale_norm = sale_norm.droplevel(THREAD)
    reward = norm_value.loc[sale_norm.index] - sale_norm
    reward = reward.reindex(index=lstgs, fill_value=0)
    return reward.mean(), (reward * start_price).mean()


def get_reward(data=None, values=None, part=None, delta=None, byr=False):
    if values is None:
        values = load_values(part=part, delta=delta)

    # restrict to valid and calculate rewards
    if byr:
        data = get_byr_valid(data)
        reward = get_byr_reward(data=data, values=values)
    else:
        data = get_slr_valid(data)
        reward = get_slr_reward(data=data, values=values, delta=delta)

    return reward


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


def load_values(part=None, delta=None):
    df = load_file(part, 'values')
    v = df.sale_price * delta ** df.relist_ct
    v = v.groupby(LSTG).mean()
    return v


def backward_from_done(x=None, done=None):
    """
    Propagates value at done across trajectory. Operations
    vectorized across all trailing dimensions after the first [T,].
    :param tensor x: tensor to propagate across trajectory
    :param tensor done: indicator for end of trajectory
    :return tensor newx: value at done at every step of trajectory
    """
    dtype = x.dtype  # cast new tensors to this data type
    T, N = x.shape  # time steps, number of envs

    # recast
    done = done.type(torch.int)

    # initialize output tensor
    newx = torch.zeros(x.shape, dtype=dtype)

    # vector for given time period
    v = torch.zeros(N, dtype=dtype)

    for t in reversed(range(T)):
        v = v * (1 - done[t]) + x[t] * done[t]
        newx[t] += v

    return newx


def valid_from_done(done):
    """Returns a float mask which is zero for all time-steps after the last
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved."""
    done = done.type(torch.float)
    done_count = torch.cumsum(done, dim=0)
    done_max, _ = torch.max(done_count, dim=0)
    valid = torch.abs(done_count - done_max) + done
    valid = torch.clamp(valid, max=1)
    return valid.bool()
