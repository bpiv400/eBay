import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils import topickle, load_data
from constants import PLOT_DIR, DAY
from assess.const import NORM1_DIM_SHORT
from featnames import DAYS_SINCE_LSTG, X_OFFER, X_THREAD, INDEX, NORM, THREAD, \
    LOOKUP, START_PRICE, ACC_PRICE, DEC_PRICE, START_TIME, END_TIME


def get_feats(data=None):
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


def main():
    d = dict()

    # data
    data = load_data()

    # interarrival waiting time
    y, x1, x2 = get_feats(data)

    xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, np.linspace(.5, 3, 50))
    mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

    idx = pd.MultiIndex.from_tuples(list(mesh), names=['norm1', 'days'])
    out = minimize(loss(y=y, x1=x1, x2=x2), x0=np.zeros(4))
    y_hat = 1 / get_lambda(params=out.x, x1=mesh[:, 0], x2=mesh[:, 1])

    d['contour_interarrival'] = pd.Series(y_hat, index=idx)

    # save
    topickle(d, PLOT_DIR + 'obs.pkl')


if __name__ == '__main__':
    main()
