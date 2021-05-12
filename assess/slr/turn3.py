import numpy as np
import pandas as pd
from assess.util import ll_wrapper, add_byr_reject_on_lstg_end, kreg2
from utils import topickle, load_data, safe_reindex
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM, POINTS, NORM1_BIN_MESH
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, EXP, AUTO, NORM, LOOKUP, START_PRICE


def rejection_plot(y=None, x1=None, x2=None):
    cols, df = {'Active': 1., 'Expiration': 1.1, 'Automatic': 1.05}, None
    for k, v in cols.items():
        mask = np.isclose(x2, v)
        line, bw = ll_wrapper(y[mask], x1[mask], dim=NORM1_DIM)
        if df is None:
            line.columns = pd.MultiIndex.from_product([[k], line.columns])
            df = line
        else:
            for col in ['beta', 'err']:
                df.loc[:, (k, col)] = line[col]
        print('{}: {}'.format(k, bw[0]))
    return df


def main():
    d = dict()

    data = load_data()
    con = add_byr_reject_on_lstg_end(con=data[X_OFFER][CON])

    con3 = con.xs(3, level=INDEX)
    norm1 = data[X_OFFER][NORM].xs(1, level=INDEX).loc[con3.index]
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE],
                                        idx=con3.index))

    df2 = data[X_OFFER][[NORM, AUTO, EXP]].xs(2, level=INDEX).loc[con3.index]
    norm2 = (1 - df2[NORM]) + .05 * df2[AUTO] + .1 * df2[EXP]
    assert norm2.max() == 1.1

    x1, x2, x3 = norm1.values, norm2.values, log10_price.values
    y_con = con3.values
    y_acc, y_rej = (y_con == 1), (y_con == 0)

    # turn1 vs. turn3 after turn2 reject
    for feat in ['acc', 'rej']:
        key = 'simple_rej{}'.format(feat)
        y = locals()['y_{}'.format(feat)]
        d[key] = rejection_plot(y=y, x1=x1, x2=x2)

    # turn2 vs. turn3, excluding turn2 reject
    for c in COMMON_CONS[1]:
        mask = np.isclose(x1, c) & (x2 <= 1)
        for feat in ['acc', 'rej', 'con']:
            y = locals()['y_{}'.format(feat)]
            if feat == 'con':
                x = x2[mask & (y > 0) & (y < 1)]
                y = y[mask & (y > 0) & (y < 1)]
            else:
                y, x = y[mask], x2[mask]
            dim = np.linspace(np.quantile(x, .05), 1, POINTS)
            line, dots, bw = ll_wrapper(y, x, discrete=[1], dim=dim)
            print('{}_{}: {}'.format(feat, c, bw[0]))

            key = 'response_slrrej{}_{}'.format(feat, c)
            d[key] = line, dots

    # by first offer and list price
    mask = np.isclose(x2, 1)
    for feat in ['acc', 'rej']:
        y = locals()['y_{}'.format(feat)]
        d['contour_slrrejbin{}'.format(feat)], bw = \
            kreg2(y=y[mask], x1=x1[mask], x2=x3[mask], mesh=NORM1_BIN_MESH)
        print('List price, {}: {}'.format(feat, bw))

    topickle(d, PLOT_DIR + 'slr3.pkl')


if __name__ == '__main__':
    main()
