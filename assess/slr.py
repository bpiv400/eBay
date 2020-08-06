import numpy as np
import pandas as pd
from agent.util import get_run_id
from assess.util import load_data, get_pctiles, discrete_pdf
from utils import topickle, load_file
from agent.const import DELTA
from constants import PLOT_DIR, AGENT_DIR, TEST, IDX
from featnames import LOOKUP, CON, NORM, START_PRICE, REJECT, ACCEPT, SLR


def gaussian_kernel(x, bw=.02):
    return lambda z: np.exp(-0.5 * ((x - z) / bw) ** 2)


def local_constant(y, kernel=None, dim=None):
    y_hat = np.zeros_like(dim)
    for i, z in enumerate(dim):
        k = kernel(z)
        y_hat[i] = np.sum(y * k) / np.sum(k)
    return y_hat


def slr_response(offers=None):
    idx_slr = offers.index.isin(IDX[SLR], level='index')
    norm = offers[NORM].shift().loc[idx_slr]
    con = offers.loc[idx_slr, CON]
    dim = np.arange(.4, .96, .01)
    y_hat = {}
    for t in IDX[SLR]:
        x = norm.xs(t, level='index').values
        kernel = gaussian_kernel(x)
        con_t = con.xs(t, level='index').values
        df = pd.DataFrame(0., index=dim, columns=[ACCEPT, REJECT, CON])

        # accept and reject probability
        y_accept = con_t == 1
        y_reject = con_t == 0
        df.loc[:, ACCEPT] = local_constant(y_accept, kernel=kernel, dim=dim)
        df.loc[:, REJECT] = local_constant(y_reject, kernel=kernel, dim=dim)

        # average concession
        idx_con = ~y_accept & ~y_reject
        x_con = x[idx_con]
        kernel_con = gaussian_kernel(x_con)
        y_con = con_t[idx_con]
        df.loc[:, CON] = local_constant(y_con, kernel=kernel_con, dim=dim)

        # rename columns
        df.rename({ACCEPT: 'Accept rate',
                   REJECT: 'Reject rate',
                   CON: 'Average concession'},
                  axis=1, inplace=True)

        y_hat[t] = df

    return y_hat


def num_months(threads=None):
    s = threads.reset_index('sim')['sim'].groupby(
        'lstg').max().squeeze() + 1
    return discrete_pdf(s, censoring=4)


def num_threads(threads=None):
    s = threads.iloc[:, 0].groupby('lstg').count()
    return discrete_pdf(s, censoring=4)


def num_offers(offers=None):
    s = offers.iloc[:, 0].groupby(['lstg', 'sim', 'thread']).count()
    return discrete_pdf(s)


def get_sale_pctiles(offers=None, start_price=None):
    sale_norm = offers.loc[offers[CON] == 1, NORM]
    # redefine norm to be fraction of list price
    slr_turn = (sale_norm.index.get_level_values(level='index') % 2) == 0
    sale_norm.loc[slr_turn] = 1 - sale_norm.loc[slr_turn]
    # keep only lstg in index
    sale_norm = sale_norm.reset_index(sale_norm.index.names[1:], drop=True)
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
    d['sale_norm'], d['sale_price'] = \
        get_sale_pctiles(offers=offers, start_price=start_price)
    d['num_months'] = num_months(threads)
    d['num_threads'] = num_threads(threads)
    d['num_offers'] = num_offers(offers)
    for k, v in d.items():
        d[k] = v.rename(name)

    return d


def main():
    # simulated seller
    start_price = load_file(TEST, LOOKUP)[START_PRICE]
    threads, offers, _ = load_data(part=TEST)
    lstgs = threads.index.get_level_values(level='lstg').unique()
    start_price = start_price.reindex(index=lstgs)  # drop infreq arrivals

    # initialize output dictionary with simulated seller
    d = collect_outputs(threads=threads,
                        offers=offers,
                        start_price=start_price,
                        name='Simulated seller')

    # seller responses
    y_hat = {-1: slr_response(offers)}  # delta = -1 ==> simulated seller

    # loop over delta
    log_dir = AGENT_DIR + '{}/'.format(SLR)
    for delta in DELTA:
        print(delta)
        # data from agent seller
        run_id = get_run_id(delta=delta,
                            entropy_coeff=.001)  # TODO: find best run
        run_dir = log_dir + 'run_{}/'.format(run_id)
        threads_sim, offers_sim, _ = load_data(part=TEST, run_dir=run_dir)
        name = 'RL seller: $\\delta = {}$'.format(delta)

        # dictionary of series
        d_sim = collect_outputs(threads=threads_sim,
                                offers=offers_sim,
                                start_price=start_price,
                                name=name)
        for k, v in d.items():  # put in DataFrames
            d[k] = pd.concat([v, d_sim[k]], axis=1)

        y_hat[delta] = slr_response(offers_sim)

    # fill in nans
    for k in ['sale_price', 'sale_norm']:
        d[k] = fill_nans(d[k])

    # put in one dictionary
    d['y_hat'] = y_hat

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(SLR))


if __name__ == '__main__':
    main()
