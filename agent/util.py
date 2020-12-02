import os
from utils import unpickle, load_file, load_data, get_role, safe_reindex
from constants import AGENT_DIR, IDX, SIM_DIR
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, CON, \
    LOOKUP, X_THREAD, X_OFFER, START_PRICE, ENTROPY, DELTA, DROPOUT, \
    BYR_AGENT, VALIDATION


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
    delta = float(delta)
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


def get_byr_agent(data=None):
    return data[X_THREAD][data[X_THREAD][BYR_AGENT]].index


def get_byr_valid(data=None):
    # for observed data, all listings are valid
    if BYR_AGENT not in data[X_THREAD].columns:
        idx = data[X_THREAD].xs(1, level=THREAD).index
    else:
        idx = get_byr_agent(data).droplevel(THREAD)
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=idx)
    return data


def load_valid_data(part=None, run_dir=None, byr=False):
    # error checking
    if run_dir is not None:
        if byr:
            assert BYR in run_dir
        else:
            assert SLR in run_dir

    # load data
    data = load_data(part=part, run_dir=run_dir)
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
