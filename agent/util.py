import os
import pandas as pd
from compress_pickle import load
from utils import load_file, get_cut
from agent.const import FEAT_TYPE, ENTROPY_BONUS
from constants import AGENT_DIR, BYR, SLR, POLICY_SLR, POLICY_BYR, MONTH
from featnames import LOOKUP, META, CON, NORM, START_PRICE, \
    START_TIME, BYR_HIST


def get_months(lstg_start=None, sale_time=None):
    months = (sale_time.groupby('lstg').first() - lstg_start) / MONTH
    months += sale_time.index.get_level_values(level='sim')
    return months


def get_proceeds(lookup=None, idx_sale=None, norm=None):
    cut = lookup[META].apply(get_cut)
    sale_norm = norm.loc[idx_sale]
    slr_turn = (idx_sale.get_level_values(level='index') % 2) == 0
    sale_norm.loc[slr_turn] = 1 - sale_norm.loc[slr_turn]
    sale_norm = sale_norm.groupby('lstg').first()
    sale_price = sale_norm * lookup[START_PRICE]
    proceeds = sale_price * (1 - cut)
    bin_proceeds = lookup[START_PRICE] * (1-cut)
    return proceeds, bin_proceeds


def get_idx_sale(offers=None):
    return offers[offers[CON] == 1].index


def get_values(part=None, run_dir=None, prefs=None):
    # load outcomes
    lookup = load_file(part, LOOKUP)
    clock = load(run_dir + '{}/clock.gz'.format(part))
    offers = load(run_dir + '{}/x_offer.gz'.format(part))
    if prefs.byr:
        raise NotImplementedError()
    else:
        idx_sale = get_idx_sale(offers=offers)
        proceeds, bin_proceeds = get_proceeds(lookup=lookup,
                                              idx_sale=idx_sale,
                                              norm=offers[NORM])
        months = get_months(lstg_start=lookup[START_TIME],
                            sale_time=clock.loc[idx_sale])

        raw_values = prefs.get_return(months_to_sale=months,
                                      months_since_start=0,
                                      sale_proceeds=proceeds,
                                      action_diff=0)

        max_values = prefs.get_max_return(months_since_start=0,
                                          bin_proceeds=bin_proceeds)

    norm_values = raw_values / max_values  # normalized values

    # put into dataframe
    values = pd.concat([raw_values.rename('raw'),
                        norm_values.rename('norm')], axis=1)

    return values


def save_run(log_dir=None, run_id=None, econ_params=None, kl_penalty=None):
    path = log_dir + 'runs.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.DataFrame(index=pd.Index([], name='run_id'))

    # economic parameters
    for k, v in econ_params.items():
        df.loc[run_id, k] = v

    # entropy bonus or cross-entropy penalty
    if kl_penalty is None:
        df.loc[run_id, 'entropy_coeff'] = ENTROPY_BONUS
    else:
        df.loc[run_id, 'entropy_coeff'] = kl_penalty

    # # moments of value distribution
    # for col in values.columns:
    #     s = values[col]
    #     df.loc[run_id, '{}_mean'.format(col)] = s.mean()
    #     df.loc[run_id, '{}_median'.format(col)] = s.median()
    #     df.loc[run_id, '{}_min'.format(col)] = s.min()
    #     df.loc[run_id, '{}_max'.format(col)] = s.max()
    #     df.loc[run_id, '{}_std'.format(col)] = s.std()

    # save
    df.to_csv(path, float_format='%.4f')


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def make_log_dir(agent_params=None):
    if agent_params[BYR]:
        log_dir = AGENT_DIR + '{}/hist_{}/'.format(
            BYR, agent_params[BYR_HIST])
    else:
        log_dir = AGENT_DIR + '{}/{}/'.format(
            SLR, agent_params[FEAT_TYPE])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir
