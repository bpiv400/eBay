import pandas as pd
from utils import unpickle, load_file, load_data, safe_reindex, get_role
from constants import IDX, EPS
from paths import PARTS_DIR, SIM_DIR, AGENT_DIR
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, CON, LOOKUP, X_OFFER, \
    START_PRICE, X_THREAD, OUTCOME_FEATS, TEST, SIM, IS_AGENT, CLOCK


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


def get_sim_dir(byr=None, delta=None, part=TEST, heuristic=False,
                turn_cost=0, index=None):
    if byr and heuristic:
        assert delta is None
        assert turn_cost == 0
        sim_dir = get_log_dir(byr=True) + 'heuristic/{}/{}/'.format(index, part)
    else:
        assert index is None
        run_dir = get_run_dir(byr=byr,
                              delta=delta,
                              turn_cost=turn_cost,
                              verbose=False)
        sim_dir = run_dir + '{}/'.format(part)
        if heuristic:
            sim_dir += 'heuristic/'
    print(sim_dir)
    return sim_dir


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
    if byr or (type(values) is int and values == 0):
        cont_value = pd.Series(0, index=idx_no_sale, name='vals')
    else:
        cont_value = safe_reindex(values, idx=idx_no_sale)
    return sale_norm, cont_value


def get_byr_valid(data=None):
    if SIM in data[X_THREAD].index.names:  # agent simulations
        assert IS_AGENT in data[X_THREAD].columns
        data[X_THREAD] = data[X_THREAD][data[X_THREAD][IS_AGENT]].drop(IS_AGENT, axis=1)
        agent_thread = data[X_THREAD].index
        data = safe_reindex(data, idx=agent_thread)
    if LOOKUP in data:
        data[LOOKUP] = data[LOOKUP].loc[data[X_THREAD].index].droplevel(THREAD)
    return data


def get_slr_valid(data=None):
    # count non-automatic seller offers
    auto = data[X_OFFER][AUTO]
    mask = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    mask = mask.droplevel([THREAD, INDEX])
    s = mask.groupby(mask.index.names).sum()
    valid = s[s > 0].index
    data = safe_reindex(data, idx=valid)
    for k in [X_OFFER, CLOCK]:
        if k in data:
            data[k] = data[k].reorder_levels(list(valid.names) + [THREAD, INDEX])
    return data


def load_valid_data(part=TEST, sim_dir=None, byr=None,
                    clock=False, minimal=False, lookup=True):
    # error checking
    if sim_dir is not None:
        assert byr is None
        byr = BYR in sim_dir
    assert byr is not None

    # load data
    if byr and sim_dir is None:  # synthetic data
        data = load_data(part=part, clock=clock, lookup=lookup)
        idx = unpickle(PARTS_DIR + '{}/synthetic.pkl'.format(part))
        data = safe_reindex(data, idx=idx)
    else:
        data = load_data(part=part, sim_dir=sim_dir, clock=clock, lookup=lookup)
        if X_OFFER not in data:
            return None

    # restrict columns
    if minimal:
        data[X_OFFER] = data[X_OFFER][OUTCOME_FEATS]

    # restrict to valid listings
    return get_byr_valid(data=data) if byr else get_slr_valid(data)


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
        assert v.max() <= 1 + EPS
    return v.rename('vals')
