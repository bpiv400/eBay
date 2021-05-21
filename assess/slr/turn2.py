import numpy as np
import pandas as pd
from agent.util import get_sim_dir, load_valid_data
from assess.util import ll_wrapper, kreg2
from utils import topickle, safe_reindex
from agent.const import DELTA_SLR
from assess.const import NORM1_DIM, NORM1_BIN_MESH, SLR_NAMES
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, NORM, ACCEPT, REJECT, AUTO, LOOKUP, START_PRICE

KEYS = [ACCEPT, REJECT, CON]


def get_y(con=None, key=None):
    mask = np.ones_like(con).astype(bool)
    if key == ACCEPT:
        y = con == 1
    elif key == REJECT:
        y = con == 0
    elif key == CON:
        y = con
        mask = (con > 0) & (con < 1)
    else:
        raise ValueError('Invalid key: {}'.format(key))
    return y, mask


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
    data = load_valid_data(byr=False, minimal=True)
    x, con, x2 = get_feats(data=data)

    for key in KEYS:
        y, mask = get_y(con=con, key=key)
        line, bw[key] = ll_wrapper(y=y[mask], x=x[mask], dim=NORM1_DIM)
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d['response_{}'.format(key)] = line
        print('{}: {}'.format(key, bw[key][0]))

        if key == REJECT:
            d['contour_rejbin_data'], bw2 = \
                kreg2(y=y, x1=x, x2=x2, mesh=NORM1_BIN_MESH)
            print('rejbin: {}'.format(bw2))

    # seller runs
    for delta in DELTA_SLR:
        sim_dir = get_sim_dir(byr=False, delta=delta)
        data = load_valid_data(sim_dir=sim_dir, minimal=True)
        x, con, x2 = get_feats(data=data)

        for key in KEYS:
            y, mask = get_y(con=con, key=key)
            d['response_{}'.format(key)].loc[:, (SLR_NAMES[delta], 'beta')], _ = \
                ll_wrapper(y=y[mask], x=x[mask],
                           dim=NORM1_DIM, bw=bw[key], ci=False)

            # 2D: norm1 and log10_start_price
            if key == REJECT:
                d['contour_rejbin_{}'.format(delta)], _ = \
                    kreg2(y=y, x1=x, x2=x2, mesh=NORM1_BIN_MESH, bw=bw2)

    topickle(d, PLOT_DIR + 'slr2.pkl')


if __name__ == '__main__':
    main()
