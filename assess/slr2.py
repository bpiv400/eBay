import numpy as np
import pandas as pd
from agent.util import find_best_run, load_valid_data
from assess.util import ll_wrapper, kreg2
from utils import topickle, load_data, safe_reindex
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM, DELTA_SLR
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, NORM, ACCEPT, REJECT, AUTO, \
    LOOKUP, START_PRICE

KEYS = [ACCEPT, REJECT, 'counter', NORM]


def get_y_x(feats=None, key=None):
    x = feats[0].values
    if key == ACCEPT:
        y = feats[2].values == 1
    elif key == REJECT:
        y = feats[2].values == 0
    elif key == 'counter':
        y = ((feats[2] < 1) & (feats[2] > 0)).values
    elif key == NORM:
        y = feats[1].values
    elif key == CON:
        is_counter = (feats[2] < 1) & (feats[2] > 0)
        y, x = feats[2][is_counter].values, x[is_counter]
    else:
        raise NotImplementedError('Invalid key: {}'.format(key))
    return y, x


def get_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO], [CON, NORM]]
    norm2 = 1 - df[NORM].xs(2, level=INDEX)
    con2 = df[CON].xs(2, level=INDEX)
    norm1 = df[NORM].xs(1, level=INDEX).loc[norm2.index]
    # throw out small opening concessions (helps with bandwidth estimation)
    norm1 = norm1[norm1 > .33]
    norm2 = norm2.loc[norm1.index]
    con2 = con2.loc[norm1.index]
    # log of start price
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE], idx=norm1.index))
    return norm1, norm2, con2, log10_price


def main():
    d, bw, bw2 = dict(), dict(), None

    # load data
    data = load_data(part=TEST)
    feats = get_feats(data=data)

    for key in KEYS:
        y, x = get_y_x(feats=feats, key=key)
        line, dots, bw[key] = ll_wrapper(y, x,
                                         dim=NORM1_DIM,
                                         discrete=COMMON_CONS[1])
        line.columns = pd.MultiIndex.from_product([['Data'], line.columns])
        dots.columns = pd.MultiIndex.from_product([['Data'], dots.columns])
        d['response_{}'.format(key)] = line, dots
        print('{}: {}'.format(key, bw[key][0]))

        if key == REJECT:
            d['contour_rejdata'], bw2 = kreg2(y=y, x1=x, x2=feats[3].values)

    # seller run
    run_dir = find_best_run(byr=False, delta=DELTA_SLR)
    data = load_valid_data(part=TEST, run_dir=run_dir)
    feats = get_feats(data=data)

    for key in KEYS:
        y, x = get_y_x(feats=feats, key=key)
        line, dots, _ = ll_wrapper(y, x,
                                   dim=NORM1_DIM,
                                   discrete=COMMON_CONS[1],
                                   bw=bw[key],
                                   ci=False)

        k = 'response_{}'.format(key)
        d[k][0].loc[:, ('Agent', 'beta')] = line
        d[k][1].loc[:, ('Agent', 'beta')] = dots

        # 2D: norm1 and log10_start_price
        if key == REJECT:
            d['contour_rejagent'], _ = kreg2(y=y, x1=x, x2=feats[3].values, bw=bw2)

    topickle(d, PLOT_DIR + 'slr2.pkl')


if __name__ == '__main__':
    main()
