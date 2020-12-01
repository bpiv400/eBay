import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.const import COMMON_CONS
from assess.const import OPT, NORM1_DIM, LOG10_BIN_DIM, POINTS
from assess.util import ll_wrapper
from utils import load_data, topickle, safe_reindex
from constants import PLOT_DIR
from featnames import INDEX, X_OFFER, NORM, ACCEPT, REJECT, CON, LOOKUP, \
    START_PRICE, TEST


def main():
    d = dict()

    # load data
    data = load_data(part=TEST)

    # first offer and response
    df = data[X_OFFER][[CON, NORM]]
    norm1 = df[NORM].xs(1, level=INDEX)
    norm1 = norm1[(.33 <= norm1) & (norm1 < 1)]  # remove BINs and very small offers
    norm2 = 1 - df[NORM].xs(2, level=INDEX).reindex(index=norm1.index, fill_value=0)
    norm2 = norm2.reindex(index=norm1.index)

    # by start price
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=norm1.index)
    x2 = np.log10(start_price).values

    # remove values not close to .5
    x1 = norm1.values
    mask = (.4 < x1) & (x1 < .6)
    yy = norm2.values[mask]
    xx = np.stack([x1, x2], axis=1)[mask, :]

    # exact estimate
    is50 = np.isclose(xx[:, 0], .5)
    ll = KernelReg(yy[is50], xx[is50, 1], var_type='c', defaults=OPT)

    # approximate
    ll2 = KernelReg(yy[~is50], xx[~is50, :], var_type='cc', defaults=OPT)
    dim2 = np.stack([np.repeat(0.5, POINTS), LOG10_BIN_DIM], axis=1)

    df = pd.DataFrame(index=LOG10_BIN_DIM)
    df['Approximate'] = ll2.fit(dim2)[0]
    df['Exact'] = ll.fit(LOG10_BIN_DIM)[0]

    d['response_bin'] = df

    topickle(d, PLOT_DIR + 'byr2.pkl')


if __name__ == '__main__':
    main()
