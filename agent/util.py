import os
from utils import unpickle, load_file, load_data, get_role, safe_reindex
from constants import AGENT_DIR, IDX, SIM_DIR
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, CON, TURN_COST, \
    LOOKUP, X_THREAD, X_OFFER, START_PRICE, DELTA


def get_log_dir(**kwargs):
    log_dir = AGENT_DIR + '{}/'.format(get_role(kwargs[BYR]))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_run_id(**kwargs):
    run_id = '{}'.format(kwargs[DELTA])
    if TURN_COST in kwargs and kwargs[TURN_COST] > 0:
        run_id += '_{}'.format(kwargs[TURN_COST])
    return run_id


def get_run_dir(**kwargs):
    log_dir = get_log_dir(**kwargs)
    run_id = get_run_id(**kwargs)
    run_dir = log_dir + 'run_{}/'.format(run_id)
    if 'heuristic' in kwargs and kwargs['heuristic']:
        run_dir += '{}/heuristic/'.format(kwargs['part'])
    return run_dir


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


def get_byr_agent(data=None):
    return data[X_THREAD].xs(1, level=THREAD, drop_level=False).index


def only_byr_agent(data=None, drop_thread=False):
    for k, v in data.items():
        if THREAD in v.index.names:
            data[k] = v.xs(1, level=THREAD, drop_level=drop_thread)
    return data


def get_byr_valid(data=None):
    idx = get_byr_agent(data).droplevel(THREAD)
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=idx)
    return data


def load_valid_data(part=None, run_dir=None, byr=False, lstgs=None):
    # error checking
    if run_dir is not None:
        if byr:
            assert BYR in run_dir
        else:
            assert SLR in run_dir

    # load data
    data = load_data(part=part, run_dir=run_dir, lstgs=lstgs)
    if X_OFFER not in data:
        return None

    # restrict to valid listings
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
    df = unpickle(SIM_DIR + '{}/values.pkl'.format(part))
    v = df.p * df.x / (1 - (1-df.p) * delta)
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
