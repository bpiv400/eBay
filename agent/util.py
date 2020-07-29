import os
import pandas as pd
from compress_pickle import load
from utils import load_file, get_cut
from agent.const import ENTROPY_BONUS
from constants import AGENT_DIR, BYR, SLR, POLICY_SLR, POLICY_BYR, MONTH, \
    LISTING_FEE
from featnames import LOOKUP, META, CON, NORM, START_PRICE, START_TIME


def get_months(lstg_start=None, sale_time=None):
    months = (sale_time.groupby('lstg').first() - lstg_start) / MONTH
    months += sale_time.index.get_level_values(level='sim')
    return months


def get_sale_norm(idx_sale=None, norm=None):
    sale_norm = norm.loc[idx_sale]
    slr_turn = (idx_sale.get_level_values(level='index') % 2) == 0
    sale_norm.loc[slr_turn] = 1 - sale_norm.loc[slr_turn]
    sale_norm = sale_norm.groupby('lstg').first()
    return sale_norm


def get_values(part=None, run_dir=None, prefs=None):
    # load outcomes
    lookup = load_file(part, LOOKUP)
    clock = load(run_dir + '{}/clock.gz'.format(part))
    offers = load(run_dir + '{}/x_offer.gz'.format(part))
    if prefs.byr:
        raise NotImplementedError()
    else:
        cut = lookup[META].apply(get_cut)
        idx_sale = offers[offers[CON] == 1].index
        sale_norm = get_sale_norm(idx_sale=idx_sale, norm=offers[NORM])
        num_listings = idx_sale.get_level_values(level='sim') + 1
        listing_fees = LISTING_FEE * num_listings
        proceeds = sale_norm * lookup[START_PRICE] * (1-cut) - listing_fees
        months = get_months(lstg_start=lookup[START_TIME],
                            sale_time=clock.loc[idx_sale])

        raw_values = prefs.get_return(months_to_sale=months,
                                      sale_proceeds=proceeds)

        max_values = lookup[START_PRICE] * (1-cut)

    norm_values = raw_values / max_values  # normalized values

    # put into dataframe
    values = pd.concat([raw_values.rename('raw'),
                        norm_values.rename('norm')], axis=1)

    return values


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def get_run_id(kl=None, delta=None, beta=None):
    suffix = 'delta_{}_beta_{}'.format(delta, beta)
    if kl is None:
        run_id = 'entropy_{}_{}'.format(ENTROPY_BONUS, suffix)
    else:
        run_id = 'kl_{}_{}'.format(kl, suffix)
    return run_id
