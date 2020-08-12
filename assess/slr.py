import argparse
import numpy as np
import pandas as pd
from agent.util import get_paths
from assess.util import load_data, get_pctiles, discrete_pdf
from utils import topickle, load_file
from constants import PLOT_DIR, TEST, IDX
from featnames import CON, NORM, SLR, AUTO, LOOKUP, START_PRICE

DIM = np.arange(.4, .96, .01)


def gaussian_kernel(x, bw=.02):
    return lambda z: np.exp(-0.5 * ((x - z) / bw) ** 2)


def nw(y, kernel=None, dim=DIM):
    y_hat = pd.Series(0, index=dim)
    for z in dim:
        k = kernel(z)
        y_hat.loc[z] = np.sum(y * k) / np.sum(k)
    return y_hat


def slr_action(offers=None):
    norm = offers[NORM].groupby(offers.index.names[:-1]).shift().dropna()
    y_hat = {}
    for t in IDX[SLR]:
        x = norm.xs(t, level='index')
        kernel = gaussian_kernel(x)
        con = offers[CON].xs(t, level='index')
        auto = offers[AUTO].xs(t, level='index')
        assert np.all(con.index == x.index)
        assert np.all(auto.index == x.index)
        # accept and reject probability
        df = pd.DataFrame(index=DIM)
        df['Auto reject'] = nw((con == 0) & auto, kernel=kernel)
        df['Manual reject'] = nw((con == 0) & ~auto,  kernel=kernel)
        df['Manual accept'] = nw((con == 1) & ~auto, kernel=kernel)
        df['Auto accept'] = nw((con == 1) & auto, kernel=kernel)
        df['Concession'] = 1 - df.sum(axis=1)
        y_hat[t] = df
    return y_hat


def slr_con(offers=None, name=None):
    idx = (offers[CON] > 0) & (offers[CON] < 1)
    norm = offers[NORM].groupby(offers.index.names[:-1]).shift().dropna()[idx]
    con = offers.loc[idx, CON]
    y_hat = {}
    for t in IDX[SLR]:
        x = norm.xs(t, level='index')
        y = con.xs(t, level='index')
        assert np.all(x.index == y.index)
        y_hat[t] = nw(y, kernel=gaussian_kernel(x)).rename(name)
    return y_hat


def num_months(threads=None):
    s = threads.reset_index('sim')['sim'].groupby(
        'lstg').max().squeeze() + 1
    return discrete_pdf(s, censoring=4)


def num_threads(threads=None, lstgs=None):
    s = threads.iloc[:, 0].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)
    return discrete_pdf(s, censoring=4)


def num_offers(offers=None):
    s = offers.iloc[:, 0].groupby(offers.index.names[:-1]).count()
    return discrete_pdf(s)


def get_sale_pctiles(offers=None, start_price=None):
    sale_norm = offers.loc[offers[CON] == 1, NORM]
    # redefine norm to be fraction of list price
    slr_turn = (sale_norm.index.get_level_values(level='index') % 2) == 0
    sale_norm.loc[slr_turn] = 1 - sale_norm.loc[slr_turn]
    # keep only lstg in index
    sale_norm = sale_norm.reset_index(sale_norm.index.names[1:], drop=True)
    # non-sales
    sale_norm = sale_norm.reindex(index=start_price.index, fill_value=0.)
    # multiply by start price to get sale price
    sale_price = np.round(sale_norm * start_price, decimals=2)
    # percentiles
    norm_pctile = get_pctiles(sale_norm)
    price_pctile = get_pctiles(sale_price)
    return norm_pctile, price_pctile


def fill_nans(df):
    df.loc[0., :] = 0.
    df = df.sort_index().ffill()
    return df


def collect_outputs(threads=None, offers=None, start_price=None, name=None):
    d = dict()
    d['cdf_norm'], d['cdf_price'] = \
        get_sale_pctiles(offers=offers, start_price=start_price)
    if 'sim' in threads.index.names:
        d['num_months'] = num_months(threads)
    d['num_threads'] = num_threads(threads=threads, lstgs=start_price.index)
    d['num_offers'] = num_offers(offers)
    for k, v in d.items():
        d[k] = v.rename(name)
    d['con'] = slr_con(offers, name=name)
    return d


def main():
    # flag for relisting environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--relist', action='store_true')
    relist = parser.parse_args().relist

    # start price
    start_price = load_file(TEST, LOOKUP)[START_PRICE]

    # simulated seller
    threads, offers, _ = load_data(part=TEST, relist=relist)

    # simulated seller
    d = collect_outputs(threads=threads,
                        offers=offers,
                        start_price=start_price,
                        name='Simulated seller')
    d['action'] = {'sim': slr_action(offers)}

    # RL seller
    path_args = dict(byr=False, delta=0., entropy_coeff=.001)
    if relist:
        path_args['delta'] = .995
    else:
        path_args['suffix'] = 'betacat'
    _, _, run_dir = get_paths(**path_args)
    threads_rl, offers_rl, _ = load_data(part=TEST, run_dir=run_dir)

    d_rl = collect_outputs(threads=threads_rl,
                           offers=offers_rl,
                           start_price=start_price,
                           name='RL seller')
    d['action']['rl'] = slr_action(offers_rl)

    # concatenate DataFrames
    for k, v in d.items():
        if k.startswith('cdf') or k.startswith('num'):
            d[k] = pd.concat([v, d_rl[k]], axis=1, sort=True)
        elif k == 'con':
            for t in IDX[SLR]:
                d[k][t] = pd.concat([v[t], d_rl[k][t]], axis=1, sort=True)

    # fill in nans
    for k in ['cdf_price', 'cdf_norm']:
        d[k] = fill_nans(d[k])

    # save
    suffix = 'relist' if relist else 'norelist'
    path = PLOT_DIR + '{}_{}.pkl'.format(SLR, suffix)
    topickle(d, path)


if __name__ == '__main__':
    main()
