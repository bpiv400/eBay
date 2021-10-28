import numpy as np
import pandas as pd
from scipy.optimize import minimize
from agent.util import get_sim_dir, load_valid_data
from analyze.util import kreg2, save_dict
from utils import load_data, safe_reindex
from agent.const import DELTA_SLR
from constants import DAY
from analyze.const import NORM1_DIM_SHORT
from featnames import DAYS_SINCE_LSTG, X_OFFER, X_THREAD, INDEX, NORM, THREAD, \
    LOOKUP, START_PRICE, ACC_PRICE, DEC_PRICE, START_TIME, END_TIME, AUTO, EXP, \
    CON, REJECT, STORE


def get_poisson_feats(data=None):
    acc_norm = (data[LOOKUP][ACC_PRICE] / data[LOOKUP][START_PRICE]).rename(NORM)
    dec_norm = (data[LOOKUP][DEC_PRICE] / data[LOOKUP][START_PRICE]).rename(NORM)
    offer1 = data[X_OFFER][NORM].xs(1, level=THREAD).xs(1, level=INDEX)
    valid = (offer1 < acc_norm.loc[offer1.index]) & \
            (offer1 >= dec_norm.loc[offer1.index])
    idx = valid[valid].index
    days = data[X_THREAD].loc[idx, DAYS_SINCE_LSTG]
    wide = days.unstack()[[1, 2]]
    y1 = (wide[2] - wide[1]).dropna()
    days_to_end = (data[LOOKUP][END_TIME] - data[LOOKUP][START_TIME]) / DAY
    solo = wide[wide[2].isna()].index
    y0 = wide[1].loc[solo] - days_to_end.loc[solo]
    y = pd.concat([y0, y1]).sort_index()
    x1 = offer1.loc[wide.index]
    x2 = wide[1]
    assert np.all(y.index == x1.index)
    assert np.all(y.index == x2.index)
    return y.values, x1.values, x2.values


def get_lambda(params=None, x1=None, x2=None):
    z = params[0] + params[1] * x1 + params[2] * x2 + params[3] * x1 * x2
    return np.exp(z)


def log_likelihood(params=None, y=None, x1=None, x2=None):
    lamb = get_lambda(params=params, x1=x1, x2=x2)
    arrival = y >= 0
    l1, y1 = lamb[arrival], y[arrival]
    lnl = np.sum(np.log(l1) - l1 * y1)
    l0, y0 = lamb[~arrival], -y[~arrival]
    lnl += np.sum(-l0 * y0)
    return lnl


def loss(y=None, x1=None, x2=None):
    return lambda z: -log_likelihood(params=z, y=y, x1=x1, x2=x2)


def run_poisson():
    # interarrival waiting time
    y, x1, x2 = get_poisson_feats(load_data())
    xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, np.linspace(.5, 3, 50))
    mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
    idx = pd.MultiIndex.from_tuples(list(mesh), names=['norm1', 'days'])
    out = minimize(loss(y=y, x1=x1, x2=x2), x0=np.zeros(4))
    y_hat = 1 / get_lambda(params=out.x, x1=mesh[:, 0], x2=mesh[:, 1])
    return pd.Series(y_hat, index=idx)


def get_slr_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO] & ~data[X_OFFER][EXP],
                           [CON, REJECT, NORM]].xs(1, level=THREAD)
    acc2 = df[CON].xs(2, level=INDEX) == 1
    rej2 = df[REJECT].xs(2, level=INDEX)
    norm1 = df[NORM].xs(1, level=INDEX).loc[rej2.index]
    days = data[X_THREAD][DAYS_SINCE_LSTG].xs(1, level=THREAD).loc[rej2.index]
    return acc2.values, rej2.values, norm1.values, days.values


def run_slr():
    d, bw = dict(), dict()

    # output mesh
    xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, np.linspace(.5, 6, 50))
    mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

    # observed sellers, by store
    data = load_valid_data(byr=False, minimal=True)
    store = data[LOOKUP][STORE]
    obs = {'nostore': {k: safe_reindex(v, idx=store[~store].index)
                       for k, v in data.items()},
           'store': {k: safe_reindex(v, idx=store[store].index)
                     for k, v in data.items()}}

    for k in ['store', 'nostore']:
        y_acc, y_rej, x1, x2 = get_slr_feats(data=obs[k])
        for feat in ['acc', 'rej']:
            y = locals()['y_{}'.format(feat)]
            key = 'contour_{}days_{}'.format(feat, k)
            if feat not in bw:
                d[key], bw[feat] = kreg2(y=y, x1=x1, x2=x2, mesh=mesh)
                print('{}: {}'.format(feat, bw[feat]))
            else:
                d[key], _ = kreg2(y=y, x1=x1, x2=x2, mesh=mesh, bw=bw[feat])

    # seller runs
    for delta in DELTA_SLR:
        sim_dir = get_sim_dir(byr=False, delta=delta)
        data = load_valid_data(sim_dir=sim_dir, minimal=True)
        y_acc, y_rej, x1, x2 = get_slr_feats(data=data)
        for feat in ['acc', 'rej']:
            y = locals()['y_{}'.format(feat)]
            d['contour_{}days_{}'.format(feat, delta)], _ = \
                kreg2(y=y, x1=x1, x2=x2, mesh=mesh, bw=bw[feat])

    return d


def main():
    d = run_slr()
    d['contour_interarrival'] = run_poisson()

    # save
    save_dict(d, 'interarrival')


if __name__ == '__main__':
    main()
