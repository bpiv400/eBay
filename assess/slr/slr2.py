import numpy as np
import pandas as pd
from agent.util import get_run_dir, load_valid_data, get_slr_valid
from assess.util import ll_wrapper, kreg2
from utils import topickle, load_data, safe_reindex
from agent.const import DELTA_SLR
from assess.const import NORM1_DIM, NORM1_BIN_MESH, SLR_NAMES
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, NORM, ACCEPT, REJECT, AUTO, \
    LOOKUP, START_PRICE

KEYS = [ACCEPT, REJECT, CON]


def get_y(con=None, key=None):
    if key == ACCEPT:
        return con == 1
    elif key == REJECT:
        return con == 0
    elif key == CON:
        return con


def get_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO], [CON, NORM]]
    con2 = df[CON].xs(2, level=INDEX)
    norm1 = df[NORM].xs(1, level=INDEX).loc[con2.index]
    # log of start price
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE],
                                        idx=norm1.index))
    return norm1.values, con2.values, log10_price.values


def main():
    d, bw, bw2 = dict(), dict(), None

    # load data
    data = get_slr_valid(load_data(part=TEST))
    x, con, x2 = get_feats(data=data)

    for key in KEYS:
        y = get_y(con=con, key=key)
        line, bw[key] = ll_wrapper(y, x, dim=NORM1_DIM)
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d['response_{}'.format(key)] = line
        print('{}: {}'.format(key, bw[key][0]))

        if key == REJECT:
            d['contour_rejbin_data'], bw2 = \
                kreg2(y=y, x1=x, x2=x2, mesh=NORM1_BIN_MESH)
            print('rejbin: {}'.format(bw2))

    # seller runs
    for delta in DELTA_SLR:
        run_dir = get_run_dir(delta=delta)
        data = load_valid_data(part=TEST, run_dir=run_dir)
        x, con, x2 = get_feats(data=data)

        for key in KEYS:
            y = get_y(con=con, key=key)
            k = 'response_{}'.format(key)
            d[k].loc[:, (SLR_NAMES[delta], 'beta')], _ = \
                ll_wrapper(y, x, dim=NORM1_DIM, bw=bw[key], ci=False)

            # 2D: norm1 and log10_start_price
            if key == REJECT:
                d['contour_rejbin_{}'.format(delta)], _ = \
                    kreg2(y=y, x1=x, x2=x2, mesh=NORM1_BIN_MESH, bw=bw2)

    topickle(d, PLOT_DIR + 'slr2.pkl')


if __name__ == '__main__':
    main()
