import numpy as np
import pandas as pd
from sklearn import tree
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import get_sale_norm, get_norm_reward
from utils import safe_reindex, load_pctile
from assess.const import OPT, VALUES_DIM, POINTS, LOG10_BIN_DIM
from constants import IDX, BYR, EPS, DAY, HOUR, MAX_DAYS, MAX_DELAY_TURN, \
    MAX_DELAY_ARRIVAL, INTERVAL_ARRIVAL, INTERVAL_CT_ARRIVAL
from featnames import DELAY, CON, NORM, AUTO, START_TIME, START_PRICE, LOOKUP, \
    MSG, DAYS_SINCE_LSTG, BYR_HIST, INDEX, X_OFFER, CLOCK, THREAD, X_THREAD, \
    REJECT, EXP, SLR


def continuous_pdf(s=None):
    return s.groupby(s).count() / len(s)


def continuous_cdf(s=None):
    cdf = continuous_pdf(s).cumsum()
    if cdf.index[0] > 0.:
        cdf.loc[0.] = 0.  # where cdf is 0
    return cdf.sort_index()


def discrete_pdf(s=None):
    return continuous_pdf(s.astype('int64'))


def discrete_cdf(s=None):
    cdf = discrete_pdf(s).cumsum()
    if cdf.index[0] > 0.:
        cdf.loc[0.] = 0.  # where cdf is 0
    return cdf.sort_index()


def arrival_pdf(s=None):
    pdf = discrete_pdf(s)
    pdf.index = (pdf.index // INTERVAL_ARRIVAL).astype('int64')
    pdf = pdf.groupby(pdf.index.name).sum()
    return pdf


def arrival_dist(threads=None):
    s = threads[DAYS_SINCE_LSTG] * DAY
    pdf = arrival_pdf(s)
    pdf.index = pdf.index / INTERVAL_CT_ARRIVAL
    return pdf


def interarrival_dist(threads=None):
    s = threads[DAYS_SINCE_LSTG] * DAY
    ct = s.groupby(s.index.names[:-1]).count()
    s = s[safe_reindex(ct, idx=s.index) > 1]
    s = s.groupby(s.index.names[:-1]).diff().dropna()
    pdf = arrival_pdf(s)
    pdf.index *= (INTERVAL_ARRIVAL / HOUR)
    pdf.index.name = 'hours'
    return pdf


def arrival_cdf(threads=None):
    s = threads[DAYS_SINCE_LSTG] / MAX_DAYS
    return continuous_cdf(s)


def hist_dist(threads=None):
    pc = load_pctile(name=BYR_HIST)
    pc = pc.reset_index().set_index('pctile').squeeze()
    y = pc.reindex(index=threads[BYR_HIST], method='pad')
    cdf = discrete_cdf(y)
    return cdf


def delay_dist(offers=None):
    delay = offers.loc[~offers[AUTO], DELAY]
    p = dict()
    for t in range(2, 8):
        s = delay.xs(t, level=INDEX) * MAX_DELAY_TURN
        cdf = continuous_cdf(s)
        cdf.index = cdf.index.values / HOUR
        p[t] = cdf
    return p


def con_dist(offers=None):
    con = offers.loc[~offers[AUTO] & ~offers[EXP], CON]
    p = dict()
    for t in range(1, 8):
        s = discrete_cdf(con.xs(t, level=INDEX) * 100)
        s.index /= 100
        p[t] = s
    return p


def norm_dist(offers=None):
    norm = offers.loc[~offers[REJECT] & ~offers[AUTO], NORM]
    p = {t: continuous_cdf(norm.xs(t, level=INDEX))
         for t in range(1, 8)}
    return p


def msg_dist(offers=None):
    idx_auto = offers[offers[AUTO]].index
    idx_exp = offers[offers[EXP]].index
    idx_acc = offers[offers[CON] == 1].index
    idx_rej = offers[offers.index.isin(IDX[BYR], level=INDEX)
                     & offers[REJECT]].index
    idx_drop = idx_auto.union(idx_exp).union(idx_acc).union(idx_rej)
    idx = offers.index.drop(idx_drop)
    s = offers.loc[idx, MSG]
    return s.groupby(INDEX).mean()


def get_last(x=None):
    return x.groupby(x.index.names[:-1]).shift().dropna()


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


def censor_threads(pdf=None, censor=4):
    # censor threads per listing
    pdf.loc[censor] = pdf[pdf.index >= censor].sum(axis=0)
    pdf = pdf[pdf.index <= censor]
    assert np.isclose(pdf.sum(), 1)
    # relabel index
    idx = pdf.index.astype(str).tolist()
    idx[-1] += '+'
    pdf.index = idx
    return pdf


def thread_number(threads):
    s = threads.reset_index()[THREAD]
    pdf = s.groupby(s).count() / len(s)
    pdf = censor_threads(pdf)
    return pdf


def num_threads(data, censor=4):
    # count threads
    s = data[X_THREAD][DAYS_SINCE_LSTG]
    s = s.groupby(s.index.names[:-1]).count()

    # add in zeros
    s = s.reindex(index=data[LOOKUP].index, fill_value=0)

    # pdf
    pdf = discrete_pdf(s)

    # censor threads per listing
    pdf = censor_threads(pdf=pdf, censor=censor)

    return pdf


def num_offers(offers):
    # pdf of counts
    group = list(offers.index.names)
    group.remove(INDEX)
    walk = (offers[CON] == 0) & offers.index.isin(IDX[BYR], level=INDEX)
    con = offers.loc[~walk, CON]
    s = con.groupby(group).count()
    s = s.groupby(s).count() / len(s)
    return s


def cdf_days(data=None):
    # time to sale
    idx_sale = data[X_OFFER][data[X_OFFER][CON] == 1].index
    clock_sale = data[CLOCK].loc[idx_sale].droplevel([THREAD, INDEX])
    dur = (clock_sale - data[LOOKUP][START_TIME]) / MAX_DELAY_ARRIVAL
    assert dur.max() < 1

    # add in non-sales
    dur = dur.reindex(index=data[LOOKUP].index, fill_value=1)

    # get cdf
    cdf = continuous_cdf(dur)
    return cdf


def cdf_sale(data=None, sales=True):
    sale_norm = get_sale_norm(data[X_OFFER])

    # add in non-sales
    if not sales:
        sale_norm = sale_norm.reindex(index=data[LOOKUP].index, fill_value=0)

    # multiply by start price to get sale price
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=sale_norm.index)
    sale_price = np.round(sale_norm * start_price, decimals=2)

    # percentiles
    norm_pctile = continuous_cdf(sale_norm)
    price_pctile = continuous_cdf(sale_price)
    return norm_pctile, price_pctile


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


def calculate_row(v=None):
    beta = v.mean()
    err = 1.96 * np.sqrt(v.var() / len(v))
    return beta, err


def ll_wrapper(y, x, dim=None, discrete=(), ci=True, bw=None):
    # discrete
    dots = pd.DataFrame()
    mask = np.ones_like(y).astype(np.bool)
    for val in discrete:
        isval = np.isclose(x, val)
        if isval.sum() > 0:
            beta, err = calculate_row(y[isval])
            dots.loc[val, 'beta'] = beta
            if ci:
                dots.loc[val, 'err'] = err
            mask[isval] = False

    # continuous
    if dim is None:
        dim = get_dim(x[mask])
    line = pd.DataFrame(index=dim)

    if bw is None:
        line['beta'], bw = local_linear(y[mask], x[mask], dim=dim)
    else:
        line['beta'], _ = local_linear(y[mask], x[mask], dim=dim, bw=bw)

    # bootstrap confidence intervals
    if ci:
        line['err'] = 1.96 * bootstrap_se(y, x, dim=dim, bw=bw)
    else:
        line, dots = line.squeeze(), dots.squeeze()

    if len(discrete) == 0:
        return line, bw
    else:
        return line, dots, bw


def transform(x):
    z = np.clip(x, EPS, 1 - EPS)
    return np.log(z / (1 - z))


def kdens(x, dim=None):
    f = KDEUnivariate(transform(x))
    f.fit(kernel='gau', bw='silverman', fft=True)
    f_hat = f.evaluate(transform(dim))
    return f_hat


def kdens_wrapper(dim=VALUES_DIM, **kwargs):
    df = pd.DataFrame(index=dim)
    for k, v in kwargs.items():
        df[k] = kdens(v, dim=dim)
    df /= df.max().max()
    return df


def get_dim(x, low=.05, high=.95):
    lower = np.quantile(x, low)
    upper = np.quantile(x, high)
    dim = np.linspace(lower, upper, POINTS)
    return dim


def get_mesh(x1=None, x2=None):
    dim1, dim2 = get_dim(x1, low=.1, high=.9), get_dim(x2, low=.1, high=.9)
    xx1, xx2 = np.meshgrid(dim1, dim2)
    mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
    return mesh


def kreg2(y=None, x1=None, x2=None, names=None, mesh=None, bw=None):
    x = np.stack([x1, x2], axis=1)
    if bw is None:
        ll2 = KernelReg(y, x, var_type='cc', defaults=OPT)
    else:
        ll2 = KernelReg(y, x, var_type='cc', bw=bw)
    if mesh is None:
        mesh = get_mesh(x1, x2)
    y_hat = ll2.fit(mesh)[0]
    idx = pd.MultiIndex.from_arrays([mesh[:, 0], mesh[:, 1]])
    if names is not None:
        idx.names = names
    s = pd.Series(y_hat, index=idx)
    return s, ll2.bw


def add_byr_reject_on_lstg_end(con=None):
    sale = (con == 1).groupby(con.index.names[:-2]).max()
    s = con.reset_index(INDEX)[INDEX]
    slr_last = s.groupby(s.index.names).max().apply(lambda x: x in IDX[SLR])
    idx = slr_last[slr_last & ~safe_reindex(sale, idx=slr_last.index)].index
    con = con.unstack()
    for t in [3, 5, 7]:
        tochange = con[t].loc[idx][con[t].loc[idx].isna()].index
        con.loc[tochange, t] = 0.
        idx = idx.drop(tochange)
    con = con.stack()
    return con


def create_cdfs(elem):
    elem = {k: continuous_cdf(v) for k, v in elem.items()}
    return pd.DataFrame(elem)


def estimate_tree(X=None, y=None, max_depth=1, criterion='entropy'):
    assert np.all(X.index == y.index)
    dt = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf = dt.fit(X.values, y.values)
    r = tree.export_text(clf, feature_names=list(X.columns))
    print(r)
    return r


def bin_vs_reward(data=None, values=None, byr=False, bw=None):
    reward = pd.concat(get_norm_reward(data=data, values=values, byr=byr))
    y = reward.reindex(index=data[LOOKUP].index).values
    x = np.log10(data[LOOKUP][START_PRICE].values)
    line, bw = ll_wrapper(y=y,
                          x=x,
                          dim=LOG10_BIN_DIM,
                          bw=bw,
                          ci=(bw is None))
    return line, bw


def get_total_con(data=None):
    norm = data[X_OFFER][NORM].unstack()
    norm.loc[:, IDX[SLR]] = 1 - norm.loc[:, IDX[SLR]]
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=norm.index)
    for t in norm.columns:
        norm[t] *= start_price
    con = pd.DataFrame(index=norm.index)
    con[1] = norm[1]
    con[2] = start_price - norm[2]
    for t in range(3, 8):
        con[t] = np.abs(norm[t - 2] - norm[t])
    for t in con.columns:
        con[t] /= start_price
    return con.fillna(0)
