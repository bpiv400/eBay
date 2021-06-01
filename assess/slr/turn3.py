import numpy as np
import pandas as pd
from assess.util import ll_wrapper, kreg2, save_dict
from utils import load_data, safe_reindex
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM, POINTS, NORM1_BIN_MESH
from constants import IDX
from featnames import X_OFFER, CON, INDEX, EXP, AUTO, NORM, LOOKUP, START_PRICE, SLR


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


def add_byr_reject_on_lstg_end(con=None):
    sale = (con == 1).groupby(con.index.names[:-2]).max()
    s = con.reset_index(INDEX)[INDEX]
    slr_last = s.groupby(s.index.names).max().apply(lambda x: x in IDX[SLR])
    idx = slr_last[slr_last & ~safe_reindex(sale, idx=slr_last.index)].index
    con = con.unstack()
    for t in [3, 5, 7]:
        tochange = con[t].loc[idx][con[t].loc[idx].isna()].index
        con.loc[tochange, t] = 0.
        idx = idx.drop(tochange)
    con = con.stack()
    return con


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
    y_acc, y_rej = (con3 == 1).values, (con3 == 0).values

    # turn1 vs. turn3 after turn2 reject
    for feat in ['acc', 'rej']:
        key = 'simple_rej{}'.format(feat)
        y = locals()['y_{}'.format(feat)]
        d[key] = rejection_plot(y=y, x1=x1, x2=x2)

    # turn2 vs. turn3, excluding turn2 reject
    for c in COMMON_CONS[1]:
        mask = np.isclose(x1, c) & (x2 <= 1)
        y, x = y_acc[mask], x2[mask]
        dim = np.linspace(np.quantile(x, .05), 1, POINTS)
        line, dots, bw = ll_wrapper(y, x, discrete=[1], dim=dim)
        print('acc_{}: {}'.format(c, bw[0]))

        key = 'response_slrrejacc_{}'.format(c)
        d[key] = line, dots

    # by first offer and list price
    mask = np.isclose(x2, 1)
    d['contour_slrrejbinacc'], bw = \
        kreg2(y=y_acc[mask], x1=x1[mask], x2=x3[mask], mesh=NORM1_BIN_MESH)
    print('List price, acc: {}'.format(bw))

    # save
    save_dict(d, 'slr3')


if __name__ == '__main__':
    main()
