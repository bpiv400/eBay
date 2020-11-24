import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils import topickle, load_data
from constants import PLOT_DIR, DAY
from assess.const import DAYS_DIM
from featnames import DAYS_SINCE_LSTG, X_OFFER, X_THREAD, TEST, INDEX, \
    NORM, THREAD, LOOKUP, START_PRICE, ACC_PRICE, DEC_PRICE, START_TIME, END_TIME

NORM1 = [.5, .67, .8]
NPOLY = 3


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
    x1 = wide[1]
    x2 = offer1.loc[wide.index]
    assert np.all(y.index == x1.index)
    assert np.all(y.index == x2.index)
    return y.values, x1.values, x2.values


def get_lambda(params=None, x=None):
    z = params[0]
    for i in range(1, len(params)):
        z += params[i] * (x ** i)
    return np.exp(z)


def log_likelihood(params=None, y=None, x=None):
    lamb = get_lambda(params=params, x=x)
    arrival = y >= 0
    l1, y1 = lamb[arrival], y[arrival]
    lnl = np.sum(np.log(l1) - l1 * y1)
    l0, y0 = lamb[~arrival], -y[~arrival]
    lnl += np.sum(-l0 * y0)
    return lnl


def loss(y=None, x=None):
    return lambda z: -log_likelihood(params=z, y=y, x=x)


def bootstrap_se(y, x, n_boot=100):
    n = len(y)
    df = pd.DataFrame(index=DAYS_DIM, columns=range(n_boot))
    for b in range(n_boot):
        idx = np.random.choice(range(n), n, replace=True)
        y_b, x_b = y[idx], x[idx]
        out = minimize(loss(y=y_b, x=x_b), x0=np.zeros(NPOLY))
        df.loc[:, b] = 1 / get_lambda(params=out.x, x=DAYS_DIM)
    return df.std(axis=1)


def main():
    # data
    data = load_data(part=TEST)

    # interarrival waiting time
    y, x1, x2 = get_feats(data)
    df = pd.DataFrame(index=pd.Index(DAYS_DIM, name='days'),
                      columns=pd.MultiIndex.from_product([NORM1, ['beta', 'err']]))
    for val in NORM1:
        mask = np.isclose(x2, val)
        out = minimize(loss(y=y[mask], x=x1[mask]), x0=np.zeros(NPOLY))
        df.loc[:, (val, 'beta')] = 1 / get_lambda(params=out.x, x=DAYS_DIM)
        df.loc[:, (val, 'err')] = bootstrap_se(y=y[mask], x=x1[mask])

    d = {'simple_interarrival': df}

    # save
    topickle(d, PLOT_DIR + 'obs.pkl')


if __name__ == '__main__':
    main()
