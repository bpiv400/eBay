import numpy as np
import pandas as pd
from agent.util import find_best_run, load_valid_data, get_slr_valid
from assess.util import ll_wrapper, kreg2
from utils import topickle, load_data, safe_reindex
from agent.const import DELTA_SLR
from assess.const import NORM1_DIM, NORM1_BIN_MESH
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, NORM, ACCEPT, REJECT, AUTO, \
    LOOKUP, START_PRICE

KEYS = [ACCEPT, REJECT, 'counter']


def get_y_x(feats=None, key=None):
    x = feats[0].values
    if key == ACCEPT:
        y = feats[1].values == 1
    elif key == REJECT:
        y = feats[1].values == 0
    elif key == 'counter':
        y = ((feats[1] < 1) & (feats[1] > 0)).values
    else:
        raise NotImplementedError('Invalid key: {}'.format(key))
    return y, x


def get_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO], [CON, NORM]]
    con2 = df[CON].xs(2, level=INDEX)
    norm1 = df[NORM].xs(1, level=INDEX).loc[con2.index]
    # throw out small opening concessions (helps with bandwidth estimation)
    norm1 = norm1[norm1 > .33]
    con2 = con2.loc[norm1.index]
    # log of start price
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE],
                                        idx=norm1.index))
    return norm1, con2, log10_price


def bin_plot(y=None, x1=None, x2=None, bw=None):
    mask = x1 < .9
    s, bw = kreg2(y=y[mask], x1=x1[mask], x2=x2[mask],
                  mesh=NORM1_BIN_MESH, bw=bw)
    print('bin: {}'.format(bw))
    return s, bw


def main():
    d, bw, bw2 = dict(), dict(), None

    # load data
    data = get_slr_valid(load_data(part=TEST))
    feats = get_feats(data=data)

    for key in KEYS:
        y, x = get_y_x(feats=feats, key=key)
        line, bw[key] = ll_wrapper(y, x, dim=NORM1_DIM)
        line.columns = pd.MultiIndex.from_product([['Data'], line.columns])
        d['response_{}'.format(key)] = line
        print('{}: {}'.format(key, bw[key][0]))

        if key == REJECT:
            d['contour_rejbin_data'], bw2 = \
                bin_plot(y=y, x1=x, x2=feats[2].values)

    # seller runs
    for delta in DELTA_SLR[:-1]:
        run_dir = find_best_run(byr=False, delta=delta)
        data = load_valid_data(part=TEST, run_dir=run_dir)
        feats = get_feats(data=data)

        for key in KEYS:
            y, x = get_y_x(feats=feats, key=key)
            line, _ = ll_wrapper(y, x, dim=NORM1_DIM, bw=bw[key], ci=False)

            k = 'response_{}'.format(key)
            col = '$\\delta = {}$'.format(delta)
            d[k].loc[:, (col, 'beta')] = line

            # 2D: norm1 and log10_start_price
            if key == REJECT and delta < 1:
                d['contour_rejbin_{}'.format(delta)], _ = \
                    bin_plot(y=y, x1=x, x2=feats[2].values, bw=bw2)

    topickle(d, PLOT_DIR + 'slr2.pkl')


if __name__ == '__main__':
    main()
