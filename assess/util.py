import argparse
import numpy as np
import pandas as pd
from compress_pickle import load
from agent.util import get_log_dir
from utils import unpickle, load_file
from assess.const import SPLITS
from constants import PARTS_DIR, IDX, MONTH, BYR, EPS, COLLECTIBLES, TEST
from featnames import EXP, DELAY, CON, NORM, AUTO, START_TIME, STORE, SLR_BO_CT, \
    START_PRICE, META, LOOKUP


def get_pctiles(s):
    n = len(s.index)
    # create series of index name and values pctile
    idx = pd.Index(np.sort(s.values), name=s.name)
    pctiles = pd.Series(np.arange(1, n+1) / n,
                        index=idx, name='pctile')
    pctiles = pctiles.groupby(pctiles.index).max()
    return pctiles


def discrete_pdf(s, censoring=None):
    s = s.groupby(s).count() / len(s)
    # censor
    if censoring is not None:
        s.loc[censoring] = s[s.index >= censoring].sum(axis=0)
        s = s[s.index <= censoring]
        assert np.abs(s.sum() - 1) < 1e-8
        # relabel index
        idx = s.index.astype(str).tolist()
        idx[-1] += '+'
        s.index = idx
    return s


def load_data(part=None, lstgs=None, obs=False, sim=False, run_dir=None):
    # folder of simulation output
    if obs:  # using data
        assert not sim
        assert run_dir is None
        folder = PARTS_DIR + '{}/'.format(part)
    elif sim:
        assert run_dir is None
        folder = PARTS_DIR + '{}/sim/'.format(part)
    else:
        folder = run_dir + '{}/'.format(part)
    # load dataframes
    data = dict()
    data['threads'] = load(folder + 'x_thread.gz')
    data['offers'] = load(folder + 'x_offer.gz')
    data['clock'] = load(folder + 'clock.gz')
    # drop censored offers
    drop = data['offers'][EXP] & (data['offers'][DELAY] < 1)
    for k in ['offers', 'clock']:
        data[k] = data[k][~drop]
    assert (data['clock'].xs(1, level='index').index == data['threads'].index).all()
    # restrict to lstgs
    for k, v in data.items():
        data[k] = v.reindex(index=lstgs, level='lstg')
    return data


def gaussian_kernel(x, bw=.02):
    return lambda z: np.exp(-0.5 * ((x - z) / bw) ** 2)


def nw(y, kernel=None, dim=None):
    y_hat = pd.Series(index=dim)
    for z in dim:
        k = kernel(z)
        v = np.sum(y * k) / np.sum(k)
        y_hat.loc[z] = v
    return y_hat


def get_sale_norm(offers=None):
    sale_norm = offers.loc[offers[CON] == 1, NORM]
    # redefine norm to be fraction of list price
    slr_turn = (sale_norm.index.get_level_values(level='index') % 2) == 0
    sale_norm = slr_turn * (1. - sale_norm) + (1. - slr_turn) * sale_norm
    # keep only lstg in index
    sale_norm = sale_norm.reset_index(sale_norm.index.names[1:], drop=True)
    return sale_norm


def get_valid_slr(auto):
    # count non-automatic seller offers in turn 2
    s = (~auto.xs(2, level='index')).groupby('lstg').sum()
    return s[s > 0].index


def get_value_slr(offers=None, start_price=None):
    norm = get_sale_norm(offers)
    valid = get_valid_slr(offers[AUTO])
    norm = norm.reindex(index=valid, fill_value=0.)
    price = norm * start_price
    return norm.mean(), price.mean()


def action_dist(offers=None, dims=None):
    norm = offers[NORM].groupby(offers.index.names[:-1]).shift().dropna()
    y_hat = {}
    for t in range(2, 8):
        # inputs
        x = norm.xs(t, level='index')
        con = offers[CON].xs(t, level='index')
        assert (con.index == x.index).all()
        if t in IDX[BYR]:
            x_plus = x[x > 0]
            con_zero = con.reindex(index=x[x == 0.].index)
            con_plus = con.reindex(index=x_plus.index)
        else:
            x_plus = x
            con_plus = con
            con_zero = None
        kernel = gaussian_kernel(x_plus)

        # reject, concession, and accept probabilities
        df = pd.DataFrame(index=dims[t])
        df['Reject'] = nw(y=(con_plus == 0), kernel=kernel, dim=dims[t])
        if t < 7:
            for i in range(len(SPLITS) - 1):
                low, high = SPLITS[i], SPLITS[i + 1]
                y = (con_plus > low) & (con_plus <= high)
                k = '{}-{}% concession'.format(int(low * 100), int(high * 100))
                df[k] = nw(y=y, kernel=kernel, dim=dims[t])
        df['Accept'] = 1 - df.sum(axis=1)

        # when seller has not conceded
        if t in IDX[BYR]:
            df.loc[0., 'Reject'] = (con_zero == 0).mean()
            if t < 7:
                for i in range(len(SPLITS) - 1):
                    low, high = SPLITS[i], SPLITS[i + 1]
                    k = '{}-{}% concession'.format(int(low * 100), int(high * 100))
                    df.loc[0., k] = ((con_zero > low) & (con_zero <= high)).mean()
            df.loc[0., 'Accept'] = (con_zero == 1).mean()
            assert abs(df.loc[0].sum() - 1.) < EPS

        # sort and put in dictionary
        y_hat[t] = df.sort_index()

    return y_hat


def get_dims(offers=None):
    norm = offers[NORM].groupby(offers.index.names[:-1]).shift().dropna()
    dim = dict()
    for t in range(2, 8):
        x = norm.xs(t, level='index')
        if t in IDX[BYR]:
            x = x[x > 0]
        low = np.ceil(np.percentile(x, 10) * 20) / 20
        high = np.floor(np.percentile(x, 90) * 20) / 20
        dim[t] = np.linspace(low, high, 100)
    return dim


def get_action_dist(data=None, dim_key=None):
    """
    Wrapper function for calling action_dist when sharing x dimension across
    multiple datasets.
    :param dict data: contains dictionaries of dataframes
    :param str dim_key: data[dim_key]['offers'] used to calculate master dimensions
    :return: dict of dataframes
    """
    dims = get_dims(data[dim_key]['offers'])
    d = {k: action_dist(offers=v['offers'], dims=dims) for k, v in data.items()}
    return d


def find_best_run(byr=None):
    log_dir = get_log_dir(byr=byr)
    df = unpickle(log_dir + 'runs.pkl')
    run_id = df['price'].idxmax()
    return log_dir + '{}/'.format(run_id)


def fill_nans(df):
    df.loc[0., :] = 0.
    df = df.sort_index().ffill()
    return df


def count_dist(df=None, level=None, valid=None):
    # reindex using valid listings
    if valid is not None:
        df = df.reindex(index=valid, level='lstg')

    # level-specific parameters
    if level == 'threads':
        group = df.index.names[0]
        censoring = 4
    elif level == 'offers':
        group = df.index.names[:-1]
        censoring = None
    else:
        raise NotImplementedError()

    # count by level
    s = df.iloc[:, 0].groupby(group).count()

    # convert counts to pdf and return
    return discrete_pdf(s, censoring=censoring)


def cdf_months(offers=None, clock=None, lookup=None, valid=None):
    idx_sale = offers[offers[CON] == 1].index
    clock_sale = clock.loc[idx_sale].reset_index(clock.index.names[1:], drop=True)
    months = (clock_sale - lookup[START_TIME]) / MONTH
    months = months.reindex(index=valid)
    assert months.max() < 1.
    months[months.isna()] = 1.
    pctile = get_pctiles(months)
    return pctile


def cdf_sale(offers=None, start_price=None, valid=None):
    if valid is None:
        valid = start_price.index

    sale_norm = get_sale_norm(offers)

    # restrict / add in 0's
    sale_norm = sale_norm.reindex(index=valid, fill_value=0.)

    # multiply by start price to get sale price
    sale_price = np.round(sale_norm * start_price.reindex(index=valid),
                          decimals=2)

    # percentiles
    norm_pctile = get_pctiles(sale_norm)
    price_pctile = get_pctiles(sale_price)
    return norm_pctile, price_pctile


def get_lookup(prefix=None):
    # subset from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str)
    subset = parser.parse_args().subset

    # restrict listings
    lookup = load_file(TEST, LOOKUP)
    if subset is not None:
        filename = '{}_{}'.format(prefix, subset)
        if subset == 'store':
            lookup = lookup[lookup[STORE]]
        elif subset == 'no_store':
            lookup = lookup[~lookup[STORE]]
        elif subset == 'exp_high':
            pc75 = np.percentile(lookup[SLR_BO_CT], 75)
            lookup = lookup[lookup[SLR_BO_CT] >= pc75]
        elif subset == 'exp_low':
            pc25 = np.percentile(lookup[SLR_BO_CT], 25)
            lookup = lookup[lookup[SLR_BO_CT] <= pc25]
        elif subset == 'price_low':
            lookup = lookup[lookup[START_PRICE] <= 20]
        elif subset == 'price_high':
            lookup = lookup[lookup[START_PRICE] >= 99]
        elif subset == 'collectibles':
            lookup = lookup[lookup[META].apply(lambda x: x in COLLECTIBLES)]
        elif subset == 'other':
            lookup = lookup[lookup[META].apply(lambda x: x not in COLLECTIBLES)]
        else:
            raise NotImplementedError('Unrecognized subset: {}'.format(subset))
    else:
        filename = prefix

    print('{}: {} listings'.format(filename, len(lookup)))
    return lookup, filename
