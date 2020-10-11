import argparse
import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import get_sale_norm
from utils import unpickle, load_file
from assess.const import SPLITS, OPT
from constants import IDX, BYR, EPS, COLLECTIBLES, TEST, MAX_DAYS, \
    MAX_DELAY_ARRIVAL, MAX_DELAY_TURN, INTERVAL_ARRIVAL, PCTILE_DIR, DAY, HOUR
from featnames import DELAY, CON, NORM, AUTO, START_TIME, STORE, SLR_BO_CT, \
    START_PRICE, META, LOOKUP, MSG, DAYS_SINCE_LSTG, BYR_HIST, INDEX, \
    X_OFFER, CLOCK


def continuous_pdf(y=None):
    counts = y.groupby(y).count()
    pdf = counts / counts.sum()
    return pdf


def continuous_cdf(y=None):
    cdf = continuous_pdf(y).cumsum()
    if cdf.index[0] > 0.:
        cdf.loc[0.] = 0.  # where cdf is 0
    return cdf.sort_index()


def discrete_pdf(y=None):
    return continuous_pdf(y.astype('int64'))


def discrete_cdf(y=None):
    cdf = discrete_pdf(y).cumsum()
    if cdf.index[0] > 0.:
        cdf.loc[0.] = 0.  # where cdf is 0
    return cdf.sort_index()


def arrival_dist(threads=None):
    s = threads[DAYS_SINCE_LSTG] * MAX_DELAY_ARRIVAL
    pdf = discrete_pdf(s)

    # sum over interval
    pdf.index = (pdf.index // INTERVAL_ARRIVAL).astype('int64')
    pdf = pdf.groupby(pdf.index.name).sum()
    pdf.index = pdf.index / (DAY / INTERVAL_ARRIVAL)

    return pdf


def hist_dist(threads=None):
    pc = unpickle(PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
    pc = pc.reset_index().set_index('pctile').squeeze()
    y = pc.reindex(index=threads[BYR_HIST], method='pad')
    cdf = discrete_cdf(y)
    return cdf


def delay_dist(offers=None):
    p = dict()
    for turn in range(2, 8):
        s = offers[DELAY].xs(turn, level=INDEX) * MAX_DELAY_TURN
        cdf = discrete_cdf(s)
        cdf.index = cdf.index.values / HOUR
        p[turn] = cdf
    return p


def con_dist(offers=None):
    con = offers.loc[~offers[AUTO], CON]
    p = dict()
    for turn in range(1, 8):
        s = con.xs(turn, level=INDEX) * 100
        p[turn] = discrete_cdf(s)
    return p


def msg_dist(offers=None):
    return offers[MSG].groupby('index').mean()


def get_pctiles(s):
    n = len(s.index)
    # create series of index name and values pctile
    idx = pd.Index(np.sort(s.values), name=s.name)
    pctiles = pd.Series(np.arange(1, n+1) / n,
                        index=idx, name='pctile')
    pctiles = pctiles.groupby(pctiles.index).max()
    return pctiles


def gaussian_kernel(x, bw=.25):
    if len(np.shape(x)) == 1:
        x = np.expand_dims(x, 1)
    bw *= np.std(x, axis=0)  # scale bandwidth by standard deviation
    return lambda z: np.exp(-0.5 * np.sum(((x - z) / bw) ** 2, axis=1))


def nw(y, kernel=None, dim=None):
    y_hat = pd.Series(index=dim)
    for i in range(len(dim)):
        k = kernel(dim[i])
        v = np.sum(y * k) / np.sum(k)
        y_hat.iloc[i] = v
    return y_hat


def get_last(x=None):
    return x.groupby(x.index.names[:-1]).shift().dropna()


def norm_norm(offers=None):
    last_norm = get_last(offers[NORM])
    last_auto = get_last(offers[AUTO])
    dim = np.arange(.5, .9 + EPS, .01)
    y_hat = {}
    for t in range(2, 7):
        x = last_norm.xs(t, level=INDEX)
        y = offers[NORM].xs(t, level=INDEX)
        assert np.all(y.index == x.index)
        if t % 2 == 0:
            y = 1 - y
        else:
            x = 1 - x
        if np.std(x) > .01:
            kernel = gaussian_kernel(x)
            y_hat[t] = nw(y, kernel=kernel, dim=dim)
            if t % 2 == 1:
                auto_t = last_auto.xs(t, level=INDEX)
                y_hat[t].loc[1.] = y[(x == 1) & ~auto_t].mean()
                y_hat[t].loc[1.05] = y[(x == 1) & auto_t].mean()
    return y_hat


def get_bounds(df, interval=.05, num=50):
    assert len(df.columns) == 2
    dims = []
    for col in df.columns:
        s = df[col]
        lower = np.floor(np.percentile(s, 10) / interval) * interval
        upper = np.ceil(np.percentile(s, 90) / interval) * interval
        dim = np.linspace(lower, upper, num)
        dims.append(dim)
    return dims


def dim_from_df(df):
    dim1, dim2 = get_bounds(df)
    X, Y = np.meshgrid(dim1, dim2)
    dims = [np.reshape(v, -1) for v in [X, Y]]
    dim = pd.MultiIndex.from_arrays(dims, names=['last_norm', 'log_price'])
    return dim


def accept3d(offers=None, other=None):
    last_norm = get_last(offers[NORM])
    df = last_norm.to_frame().join(other)

    y_hat = {}
    for t in [2, 4, 6]:
        x = df.xs(t, level='index')
        y = (offers[CON] == 1).xs(t, level='index')
        assert np.all(x.index == y.index)
        y_hat[t] = nw(y=y.values,
                      kernel=gaussian_kernel(x.values),
                      dim=dim_from_df(x))
    return y_hat


def action_dist(offers=None, dims=None):
    norm = get_last(offers[NORM])
    y_hat = {}
    for t in dims.keys():
        # inputs
        x = norm.xs(t, level='index')
        con = offers[CON].xs(t, level='index')
        assert np.all(con.index == x.index)
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
                    df.loc[0., k] = np.mean((con_zero > low) & (con_zero <= high))
            df.loc[0., 'Accept'] = (con_zero == 1).mean()
            assert abs(df.loc[0].sum() - 1.) < EPS

        # sort and put in dictionary
        y_hat[t] = df.sort_index()

    return y_hat


def get_dims(offers=None, byr=None):
    norm = offers[NORM].groupby(offers.index.names[:-1]).shift().dropna()
    dim = dict()
    if byr is None:
        turns = range(2, 8)
    elif byr:
        turns = [3, 5, 7]
    else:
        turns = [2, 4, 6]
    for t in turns:
        x = norm.xs(t, level='index')
        if t in IDX[BYR]:
            x = x[x > 0]
        low = np.ceil(np.percentile(x, 10) * 20) / 20
        high = np.floor(np.percentile(x, 90) * 20) / 20
        dim[t] = np.linspace(low, high, 100)
    return dim


def get_action_dist(offers_dim=None, offers_action=None, byr=None):
    """
    Wrapper function for calling action_dist when sharing x dimension across
    multiple datasets.
    :param DataFrame offers_dim: observed data for measuring dimension
    :param DataFrame offers_action: data from which to create action distributions
    :param bool byr: estimate buyer turns if true
    :return: dict of dataframes
    """
    dims = get_dims(offers=offers_dim, byr=byr)
    d = action_dist(offers=offers_action, dims=dims)
    return d


def concat_and_fill(v1, v2):
    df = pd.concat([v1, v2], axis=1, sort=True)
    if df.isna().sum().sum() > 0:
        df = df.sort_index().ffill()
    return df


def merge_dicts(d, d_other):
    for k, v in d.items():
        if type(v) is dict:
            for t, value in v.items():
                d[k][t] = concat_and_fill(value, d_other[k][t])
        else:
            d[k] = concat_and_fill(v, d_other[k])
    return d


def count_dist(df=None):
    # level-specific parameters
    if INDEX not in df.index.names:
        group = df.index.names[0]
        censoring = 4
    else:
        group = df.index.names[:-1]
        censoring = None

    # count by level
    s = df.iloc[:, 0].groupby(group).count()

    # convert counts to pdf
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


def cdf_days(data=None):
    idx_sale = data[X_OFFER][data[X_OFFER][CON] == 1].index
    clock_sale = data[CLOCK].loc[idx_sale].reset_index(data[CLOCK].index.names[1:], drop=True)
    days = (clock_sale - data[LOOKUP][START_TIME]) / DAY
    assert days.max() < MAX_DAYS
    days[days.isna()] = MAX_DAYS
    pctile = get_pctiles(days)
    return pctile


def cdf_sale(offers=None, start_price=None):
    sale_norm = get_sale_norm(offers)

    # restrict / add in 0's
    sale_norm = sale_norm.reindex(index=start_price.index, fill_value=0.)

    # multiply by start price to get sale price
    sale_price = np.round(sale_norm * start_price, decimals=2)

    # percentiles
    norm_pctile = get_pctiles(sale_norm)
    price_pctile = get_pctiles(sale_price)
    return norm_pctile, price_pctile


def get_lstgs(prefix=None):
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
    return lookup.index, filename


def local_linear(y, x, dim=None, bw=None):
    if bw is None:
        ll = KernelReg(y, x, var_type='c', defaults=OPT)
    else:
        ll = KernelReg(y, x, var_type='c', bw=bw)
    beta = ll.fit(dim)[0]
    return beta, ll.bw


def bootstrap_se(y, x, dim=None, bw=None, n_boot=100):
    n = len(y)
    df = pd.DataFrame(index=dim, columns=range(n_boot))
    for b in range(n_boot):
        idx = np.random.choice(range(n), n, replace=True)
        y_b, x_b = y[idx], x[idx]
        df.loc[:, b], _ = local_linear(y_b, x_b, dim=dim, bw=bw)
    return df.std(axis=1)


def ll_wrapper(y, x, dim=None, discrete=(), ci=True):
    # initialize output
    line, dots = pd.DataFrame(index=dim), pd.DataFrame()

    # discrete
    mask = np.ones_like(y).astype(np.bool)
    for val in discrete:
        isval = np.isclose(x, val)
        if isval.sum() > 0:
            y_val = y[isval]
            dots.loc[val, 'beta'] = y_val.mean()
            se = np.sqrt(y_val.var() / isval.sum())
            if ci:
                dots.loc[val, 'low'] = dots.loc[val, 'beta'] - 1.96 * se
                dots.loc[val, 'high'] = dots.loc[val, 'beta'] + 1.96 * se
            mask[isval] = False

    # continuous
    line['beta'], bw = local_linear(y[mask], x[mask], dim=dim, bw=(.05,))

    # bootstrap confidence intervals
    if ci:
        line_se = bootstrap_se(y, x, dim=dim, bw=bw)
        line['low'] = line['beta'] - 1.96 * line_se
        line['high'] = line['beta'] + 1.96 * line_se
    else:
        line, dots = line.squeeze(), dots.squeeze()

    return line, dots
