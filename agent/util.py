import pandas as pd
from utils import unpickle, load_file, load_data, safe_reindex, get_role
from constants import AGENT_DIR, IDX, SIM_DIR
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, CON, \
    LOOKUP, X_OFFER, START_PRICE, X_THREAD, OUTCOME_FEATS, TEST


def get_log_dir(byr=None):
    return AGENT_DIR + '{}/'.format(get_role(byr))


def get_run_id(byr=None, delta=None, turn_cost=0):
    if byr and turn_cost > 0:
        return '{}_{}'.format(float(delta), turn_cost)
    else:
        return '{}'.format(float(delta))


def get_run_dir(byr=None, delta=None, turn_cost=0, verbose=True):
    log_dir = get_log_dir(byr=byr)
    run_id = get_run_id(byr=byr, delta=delta, turn_cost=turn_cost)
    run_dir = log_dir + 'run_{}/'.format(run_id)
    if verbose:
        print(run_dir)
    return run_dir


def get_output_dir(byr=None, delta=None, part=TEST,
                   heuristic=False, turn_cost=0,
                   agent_thread=1, verbose=True, **kwargs):
    run_dir = get_run_dir(byr=byr,
                          delta=delta,
                          turn_cost=turn_cost,
                          verbose=False)
    output_dir = run_dir + '{}/'.format(part)
    if heuristic:
        output_dir += 'heuristic/'
    elif byr:
        output_dir += '{}/'.format(agent_thread)
    if verbose:
        print(output_dir)
    return output_dir


def get_sale_norm(offers=None, drop_thread=True):
    sale_norm = offers.loc[offers[CON] == 1, NORM]
    # redefine norm to be fraction of list price
    slr_turn = (sale_norm.index.get_level_values(level=INDEX) % 2) == 0
    sale_norm = slr_turn * (1. - sale_norm) + (1. - slr_turn) * sale_norm
    # restrict index levels
    sale_norm = sale_norm.droplevel(INDEX)
    if drop_thread and THREAD in sale_norm.index.names:
        sale_norm = sale_norm.droplevel(THREAD)
    return sale_norm


def get_norm_reward(data=None, values=None, byr=False):
    sale_norm = get_sale_norm(data[X_OFFER])
    idx_no_sale = data[LOOKUP].index.drop(sale_norm.index)
    if byr:
        sale_norm = 1 - sale_norm  # discount on list price
        assert values is None
        cont_value = pd.Series(0., index=idx_no_sale, name='vals')
    else:
        cont_value = safe_reindex(values, idx=idx_no_sale)
    return sale_norm, cont_value


def only_byr_agent(data=None, drop_thread=True, agent_thread=1):
    if data is None:
        return None

    for k, v in data.items():
        if THREAD in v.index.names:
            data[k] = v.xs(agent_thread, level=THREAD, drop_level=drop_thread)
        idx = data[X_THREAD].index
        if THREAD in idx.names:
            idx = idx.droplevel(THREAD)
        data[LOOKUP] = data[LOOKUP].loc[idx.unique()]
    return data


def get_byr_valid(data=None, agent_thread=1):
    idx = data[X_THREAD].xs(agent_thread, level=THREAD).index
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=idx)
        if k == X_OFFER:
            data[k] = data[k].reorder_levels(v.index.names)
    return data


def get_slr_valid(data=None):
    # count non-automatic seller offers
    auto = data[X_OFFER][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    valid = valid.droplevel([THREAD, INDEX])
    s = valid.groupby(valid.index.names).sum()
    lstg_sim = s[s > 0].index
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=lstg_sim)
        if k == X_OFFER:
            data[k] = data[k].reorder_levels(v.index.names)
    return data


def load_valid_data(part=TEST, run_dir=None, byr=None,
                    clock=False, minimal=False, agent_thread=1):
    # error checking
    if run_dir is not None:
        assert byr is None
        byr = BYR in run_dir

    # load data
    data = load_data(part=part, run_dir=run_dir, clock=clock)
    if X_OFFER not in data:
        return None

    # restrict columns
    if minimal:
        data[X_OFFER] = data[X_OFFER][OUTCOME_FEATS]

    # restrict to valid listings
    if byr:
        return get_byr_valid(data=data, agent_thread=agent_thread)
    else:
        return get_slr_valid(data=data)


def load_values(part=TEST, delta=None, normalize=True):
    """
    Calculates discounted value of each item in partition.
    :param str part: partition name
    :param float delta: discount parameter
    :param bool normalize: normalize by list price if True
    :return: pd.DataFrame of (normalized) values
    """
    df = unpickle(SIM_DIR + '{}/values.pkl'.format(part))
    v = df.p * df.x / (1 - (1-df.p) * delta)
    if normalize:
        start_price = load_file(part, LOOKUP)[START_PRICE]
        v /= start_price
    return v.rename('vals')
