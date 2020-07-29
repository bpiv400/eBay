import argparse
import os
import pandas as pd
from compress_pickle import load
from assess.unconditional import get_distributions
from agent.Prefs import BuyerPrefs, SellerPrefs
from agent.util import get_run_suffix
from utils import unpickle, topickle, load_file, get_cut
from constants import AGENT_DIR, BYR, SLR, TEST, MONTH, LISTING_FEE
from featnames import LOOKUP, META, CON, NORM, START_PRICE, START_TIME, \
    EXP, DELAY, AUTO


def save_values(log_dir=None, run_id=None, values=None):
    path = log_dir + 'runs.pkl'
    if os.path.isfile(path):
        df = unpickle(path)
    else:
        df = pd.DataFrame(index=pd.Index([], name='run_id'))
    # moments of value distribution
    for col in values.columns:
        s = values[col]
        df.loc[run_id, '{}_mean'.format(col)] = s.mean()
        df.loc[run_id, '{}_median'.format(col)] = s.median()
        df.loc[run_id, '{}_min'.format(col)] = s.min()
        df.loc[run_id, '{}_max'.format(col)] = s.max()
        df.loc[run_id, '{}_std'.format(col)] = s.std()
    # save
    topickle(contents=df, path=path)


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


def get_values(lookup=None, data=None, prefs=None):
    if prefs.byr:
        raise NotImplementedError()
    else:
        cut = lookup[META].apply(get_cut)
        idx_sale = data['offers'][data['offers'][CON] == 1].index
        sale_norm = get_sale_norm(idx_sale=idx_sale,
                                  norm=data['offers'][NORM])
        num_listings = idx_sale.get_level_values(level='sim') + 1
        gross = sale_norm * lookup[START_PRICE] * (1-cut)
        listing_fees = LISTING_FEE * num_listings
        proceeds = gross - listing_fees
        months = get_months(lstg_start=lookup[START_TIME],
                            sale_time=data['clock'].loc[idx_sale])

        raw_values = prefs.get_return(months_to_sale=months,
                                      sale_proceeds=proceeds)

        max_values = data[LOOKUP][START_PRICE] * (1-cut)

    norm_values = raw_values / max_values  # normalized values

    # put into dataframe
    values = pd.concat([raw_values.rename('raw'),
                        norm_values.rename('norm')], axis=1)

    return values


def load_data(part=None, run_dir=None):
    part_dir = run_dir + '{}/'.format(part)
    clock = load(part_dir + 'clock.gz')
    threads = load(part_dir + 'x_thread.gz')
    offers = load(part_dir + 'x_offer.gz')
    # drop censored and auto
    drop = offers[AUTO] | (offers[EXP] & (offers[DELAY] < 1))
    offers = offers[~drop]
    clock = clock[~drop]
    assert (clock.xs(1, level='index').index == threads.index).all()
    # put in dictionary and return
    return {'threads': threads, 'offers': offers, 'clock': clock}


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--delta', type=float)
    parser.add_argument('--beta', type=float, default=1.)
    args = parser.parse_args()
    delta, beta = args.delta, args.beta
    role = BYR if args.byr else SLR

    # lookup file
    lookup = load_file(TEST, LOOKUP)

    # create preferences
    pref_params = dict(delta=delta, beta=beta)
    pref_cls = BuyerPrefs if args.byr else SellerPrefs
    prefs = pref_cls(**pref_params)

    # collect runs
    log_dir = AGENT_DIR + '{}/'.format(role)
    suffix = get_run_suffix(delta=delta, beta=beta)
    run_dirs = [t[0] + '/' for t in os.walk(log_dir)
                if 'kl' in t[0] and t[0].endswith(suffix)]

    # loop over runs
    for run_dir in run_dirs:
        data = load_data(part=TEST, run_dir=run_dir)
        data0 = {k: v.xs(0, level='sim') for k, v in data.items()}
        p = get_distributions(threads=data0['threads'],
                              offers=data0['offers'])

        # evaluate simulations
        values = get_values(lookup=lookup, data=data, prefs=prefs)


if __name__ == '__main__':
    main()
