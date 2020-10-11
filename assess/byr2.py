import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import load_values
from agent.const import AGENT_CONS
from assess.const import OPT
from assess.util import ll_wrapper
from utils import load_data, topickle, safe_reindex
from constants import TEST, PLOT_DIR, EPS
from featnames import INDEX, X_OFFER, NORM, ACCEPT, REJECT, CON, LSTG, \
    LOOKUP, START_PRICE


def main():
    # load data
    data = load_data(part=TEST)

    # first offer and response
    norm = data[X_OFFER][NORM]
    norm1 = norm.xs(1, level=INDEX)
    norm1 = norm1[(.33 <= norm1) & (norm1 < 1)]  # remove BINs and very small offers
    norm2 = 1 - norm.xs(2, level=INDEX).reindex(index=norm1.index,
                                                fill_value=0)
    dim = np.arange(.4, 1., .01)

    # output
    d, prefix = dict(), 'response'
    for name in ['avg', ACCEPT, REJECT, CON]:
        key = '{}_{}'.format(prefix, name)
        print(key)
        x, y = norm1.values, norm2.values
        if name in [CON, 'avg']:
            if name == CON:
                mask = (y > x) & (y < 1)
                y, x = y[mask], x[mask]
        elif name == ACCEPT:
            y = y[y == x]
        else:
            y = y[y == 1]

        d[key] = ll_wrapper(y, x, dim=dim, discrete=AGENT_CONS[1])

    # # average valuation as function of slr response
    # norm2 = 1 - norm.xs(2, level=INDEX)
    # half = norm.xs(1, level=INDEX).reindex(index=norm2.index) == .5
    # norm2 = norm2[half]
    # vals = load_values(part=TEST, delta=.9).reindex(
    #     index=norm2.index, level=LSTG)
    # dim2 = np.arange(.65, .9 + EPS, .01)
    # y, x = vals.values, norm2.values
    # cont, disc = ll_wrapper(y, x, dim=dim2, discrete=[.5, 1.])

    # by start price
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=norm1.index)
    bin_dim = np.linspace(1, 3, 50)
    x2 = np.log10(start_price).values

    # remove values not close to .5
    x1 = norm1.values
    mask = (.4 < x1) & (x1 < .6)
    yy = y[mask]
    xx = np.stack([x1, x2], axis=1)[mask, :]

    # exact estimate
    is50 = np.isclose(xx[:, 0], .5)
    ll = KernelReg(yy[is50], xx[is50, 1], var_type='c', defaults=OPT)

    # approximate
    ll2 = KernelReg(yy[~is50], xx[~is50, :], var_type='cc', defaults=OPT)
    dim2 = np.stack([np.repeat(0.5, len(bin_dim)), bin_dim], axis=1)

    df = pd.DataFrame(index=bin_dim)
    df['Approximate'] = ll2.fit(dim2)[0]
    df['Exact'] = ll.fit(bin_dim)[0]

    d['{}_bin'.format(prefix)] = df

    topickle(d, PLOT_DIR + 'byr2.pkl')


if __name__ == '__main__':
    main()
