import numpy as np
import pandas as pd
from assess.util import ll_wrapper, add_byr_reject_on_listing_expiration, kreg2
from utils import topickle, load_data, safe_reindex
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM, CON2_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, EXP, AUTO, NORM, REJECT, \
    LOOKUP, START_PRICE


def rejection_plot(y=None, x1=None, x2=None):
    cols, tup = {'Active': 0, 'Expiration': -.1}, None
    for k, v in cols.items():
        mask = np.isclose(x2, v) & (x1 > .33)
        line, dots, bw = ll_wrapper(y[mask], x1[mask],
                                    discrete=COMMON_CONS[1][1:-1],
                                    dim=NORM1_DIM)
        if tup is None:
            line.columns = pd.MultiIndex.from_product([[k], line.columns])
            dots.columns = pd.MultiIndex.from_product([[k], dots.columns])
            tup = line, dots
        else:
            for col in ['beta', 'err']:
                tup[0].loc[:, (k, col)] = line[col]
                tup[1].loc[:, (k, col)] = dots[col]
        print('{}: {}'.format(k, bw[0]))
    return tup


def main():
    d = dict()

    data = load_data(part=TEST)
    con = add_byr_reject_on_listing_expiration(con=data[X_OFFER][CON])
    norm = data[X_OFFER][NORM]

    con3 = con.xs(3, level=INDEX)
    norm3 = (norm * ~data[X_OFFER][REJECT]).xs(3, level=INDEX).reindex(
        index=con3.index, fill_value=0)
    norm1 = norm.xs(1, level=INDEX).loc[con3.index]
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE], idx=con3.index))

    offers2 = data[X_OFFER][[CON, AUTO, EXP]].xs(2, level=INDEX).loc[con3.index]
    con2 = offers2[CON] - .05 * offers2[AUTO] - .1 * offers2[EXP]
    assert con2.min() == -.1

    x1, x2, x3 = norm1.values, con2.values, log10_price.values
    y_acc, y_rej, y_norm = (con3 == 1).values, (con3 == 0).values, norm3.values

    # turn1 vs. turn3 after turn2 reject
    for feat in ['acc', 'rej', 'norm']:
        key = 'response_rej{}'.format(feat)
        y = locals()['y_{}'.format(feat)]
        d[key] = rejection_plot(y=y, x1=x1, x2=x2)

    # turn2 vs. turn3, excluding turn2 reject
    for c in COMMON_CONS[1][1:-1]:
        k = '{}'.format(c)
        mask = np.isclose(x1, c) & (x2 >= 0) & (x2 <= .9)
        for feat in ['acc', 'norm']:
            key = 'response_slrrej{}'.format(feat)
            y = locals()['y_{}'.format(feat)]
            line, dots, bw = ll_wrapper(y=y[mask], x=x2[mask],
                                        discrete=[0, .5], dim=CON2_DIM)
            print('{}_{}: {}'.format(feat, c, bw[0]))

            if c == COMMON_CONS[1][1]:
                line.columns = pd.MultiIndex.from_product([[k], line.columns])
                dots.columns = pd.MultiIndex.from_product([[k], dots.columns])
                d[key] = line, dots
            else:
                for col in ['beta', 'err']:
                    d[key][0].loc[:, (k, col)] = line[col]
                    d[key][1].loc[:, (k, col)] = dots[col]

    # list price vs. turn3, after active turn2 reject or concession
    for c in COMMON_CONS[1][1:-1]:
        mask = np.isclose(x1, c) & (x2 > 0)
        s1, bw = kreg2(y=y_acc[mask], x1=x3[mask], x2=x2[mask])
        print('List price 2-D, {}: {}'.format(c, bw))
        dim = s1.index.levels[0]
        mask = np.isclose(x1, c) & np.isclose(x2, 0)
        s2, _ = ll_wrapper(y=y_acc[mask], x=x3[mask],
                           dim=dim, bw=(bw[0],), ci=False)
        for z in dim:
            idx = np.abs(s1.loc[z] - s2.loc[z]).argmin()
            s2.loc[z] = s1.loc[z].index[idx]
        d['contourline_bincon_{}'.format(c)] = s1, s2

        key = 'simple_slrrejaccbin_{}'.format(c)
        mask = np.isclose(x1, c) & np.isclose(x2, 0)
        d[key], bw = ll_wrapper(y=y_acc[mask], x=x3[mask])
        print('List price, {}: {}'.format(c, bw[0]))

    topickle(d, PLOT_DIR + 'slr3.pkl')


if __name__ == '__main__':
    main()
