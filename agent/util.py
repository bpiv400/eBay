import os
import pandas as pd
from constants import AGENT_DIR, IDX
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, LSTG, CON, \
    LOOKUP, X_THREAD, X_OFFER, START_PRICE, ENTROPY, DELTA, DROPOUT, \
    BYR_AGENT, VALIDATION
from utils import unpickle, load_file, load_data, get_role, safe_reindex


def get_log_dir(**kwargs):
    role = BYR if kwargs[BYR] else SLR
    log_dir = AGENT_DIR + '{}/delta_{}/'.format(role, kwargs[DELTA])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(**kwargs):
    # log directory
    log_dir = get_log_dir(byr=kwargs[BYR], delta=kwargs[DELTA])

    # run id
    d1, d2 = [int(elem * 10) for elem in kwargs[DROPOUT]]
    run_id = '{}_{}_{}'.format(kwargs[ENTROPY], d1, d2)
    if 'suffix' in kwargs and kwargs['suffix'] is not None:
        run_id += '_{}'.format(kwargs['suffix'])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def find_best_run(byr=None, delta=None, verbose=True):
    log_dir = get_log_dir(byr=byr, delta=delta)
    path = log_dir + '{}.pkl'.format(VALIDATION)
    if os.path.isfile(path):
        norm = unpickle(path)[NORM]
    else:
        return None
    run_id = norm[~norm.isna()].astype('float64').idxmax()
    if verbose:
        print('Best {} run for delta = {}: {}'.format(
            get_role(byr), delta, run_id))
    return log_dir + '{}/'.format(run_id)


def get_slr_valid(data=None):
    # count non-automatic seller offers
    auto = data[X_OFFER][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    valid = valid.droplevel([THREAD, INDEX])
    s = valid.groupby(valid.index.names).sum()
    lstg_sim = s[s > 0].index
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=lstg_sim)
    return data


def get_norm_reward(data=None, values=None):
    sale_norm = get_sale_norm(data[X_OFFER])
    idx_no_sale = data[LOOKUP].index.drop(sale_norm.index)
    cont_value = safe_reindex(values, idx=idx_no_sale)
    return sale_norm, cont_value


def get_slr_return(data=None, values=None):
    assert values.max() < 1
    norm = pd.concat(get_norm_reward(data=data, values=values)).sort_index()
    reward = norm * data[LOOKUP][START_PRICE]
    return norm.mean(), reward.mean()


def get_byr_agent(data=None):
    return data[X_THREAD][data[X_THREAD][BYR_AGENT]].index


def get_byr_valid(data=None):
    # for observed data, all listings are valid
    if BYR_AGENT not in data[X_THREAD].columns:
        return data

    lstg_sim = get_byr_agent(data).droplevel(THREAD)
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=lstg_sim)
    return data


def get_byr_return(data=None, values=None):
    sale_norm = get_sale_norm(data[X_OFFER], drop_thread=False)
    if BYR_AGENT in data[X_THREAD].columns:
        sale_norm = sale_norm[data[X_THREAD][BYR_AGENT]]
    start_price = data[LOOKUP][START_PRICE]
    sale_norm = sale_norm.droplevel(THREAD)
    norm_vals = safe_reindex(values, idx=sale_norm.index)
    return_norm = norm_vals - sale_norm
    return_norm = return_norm.reindex(index=data[LOOKUP].index, fill_value=0)
    return_dollar = return_norm * start_price
    return return_norm.mean(), return_dollar.mean()


def load_valid_data(part=None, run_dir=None):
    data = load_data(part=part, run_dir=run_dir)
    byr = BYR in run_dir
    if byr:
        data = get_byr_valid(data)
    else:
        data = get_slr_valid(data)
    return data


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


def load_values(part=None, delta=None, normalize=True):
    df = load_file(part, 'values')
    x = df.sale_price.groupby(LSTG).mean()
    num_sales = df.relist_ct.groupby(LSTG).count()
    num_exps = df.relist_ct.groupby(LSTG).sum()
    p = num_sales / (num_sales + num_exps)
    v = p * x / (1 - (1-p) * delta)
    if normalize:
        start_price = load_file(part, LOOKUP)[START_PRICE]
        v /= start_price
    return v.rename('vals')


def get_turn(x, byr=None):
    last = 4 * x[:, -2] + 2 * x[:, -1]
    if byr:
        return 7 - 6 * x[:, -3] - last
    else:
        return 6 - last
