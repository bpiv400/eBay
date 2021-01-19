import pandas as pd
from utils import unpickle, load_file, load_data, safe_reindex
from constants import AGENT_DIR, IDX, SIM_DIR
from featnames import SLR, BYR, NORM, AUTO, INDEX, THREAD, CON, \
    LOOKUP, X_OFFER, START_PRICE, X_THREAD


def get_run_id(delta=None):
    return '{}'.format(BYR if delta is None else float(delta))


def get_run_dir(delta=None):
    run_id = get_run_id(delta=delta)
    run_dir = AGENT_DIR + 'run_{}/'.format(run_id)
    return run_dir


def get_output_dir(delta=None, part=None, heuristic=False):
    run_dir = get_run_dir(delta=delta)
    output_dir = run_dir + '{}/'.format(part)
    if heuristic:
        output_dir += 'heuristic/'
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


def only_byr_agent(data=None, drop_thread=False):
    for k, v in data.items():
        if THREAD in v.index.names:
            data[k] = v.xs(1, level=THREAD, drop_level=drop_thread)
        idx = data[X_THREAD].index
        if THREAD in idx.names:
            idx = idx.droplevel(THREAD)
        data[LOOKUP] = data[LOOKUP].loc[idx]
    return data


def get_byr_valid(data=None, lstgs=None):
    if lstgs is None:  # find sales in data
        con = data[X_OFFER][CON]
        lstgs = con[con == 1].index.droplevel([THREAD, INDEX])

    for k, v in data.items():
        data[k] = safe_reindex(v, idx=lstgs)
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


def load_valid_data(part=None, run_dir=None, byr=None, lstgs=None):
    # error checking
    if run_dir is not None:
        assert byr is None
        byr = BYR in run_dir
    if lstgs is not None:  # for constraining simulated buyer output to sold listings in data
        assert byr
        assert run_dir is not None

    # load data
    data = load_data(part=part, run_dir=run_dir)
    if X_OFFER not in data:
        return None

    # restrict to valid listings
    if byr:
        return get_byr_valid(data=data, lstgs=lstgs)
    else:
        return get_slr_valid(data=data)


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
