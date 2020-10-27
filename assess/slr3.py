import numpy as np
import pandas as pd
from assess.util import ll_wrapper, kreg2
from utils import topickle, load_data
from assess.const import NORM1_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, EXP, AUTO, NORM


def rejection_plot(y=None, x1=None, x2=None):
    cols = {'Automatic': -.05, 'Manual': 0, 'Expiration': -.1}
    dfs = {}
    for k, v in cols.items():
        mask = np.isclose(x2, v) & (x1 > .33)
        line, bw = ll_wrapper(y[mask], x1[mask], dim=NORM1_DIM)
        dfs[k] = line
        print('{}: {}'.format(k, bw[0]))
    return pd.concat(dfs, axis=1)


def main():
    d = dict()

    data = load_data(part=TEST)

    offers2 = data[X_OFFER][[CON, AUTO, EXP]].xs(2, level=INDEX)
    offers2 = offers2[offers2[CON] < 1]
    con2 = offers2[CON]
    con2[offers2[AUTO]] = -.05
    con2[offers2[EXP]] = -.1

    norm1 = data[X_OFFER][NORM].xs(1, level=INDEX).loc[con2.index]
    con3 = data[X_OFFER][CON].xs(3, level=INDEX).reindex(
        index=con2.index, fill_value=0)
    norm3 = data[X_OFFER][NORM].xs(3, level=INDEX).reindex(index=con2.index)
    norm3[norm3.isna()] = norm1[norm3.isna()]

    x_norm1, x_con2 = norm1.values, con2.values
    y_con, y_norm = con3.values, norm3.values

    d['simple_rejacc'] = rejection_plot(y=(y_con == 1), x1=x_norm1, x2=x_con2)
    d['simple_rejrej'] = rejection_plot(y=(y_con == 0), x1=x_norm1, x2=x_con2)
    d['simple_rejnorm'] = rejection_plot(y=y_norm, x1=x_norm1, x2=x_con2)

    # contour plot for when seller does not reject
    mask = x_con2 > 0
    d['contour_delayacc'] = kreg2(y=(y_con[mask] == 1),
                                  x1=x_norm1[mask],
                                  x2=x_con2[mask])

    topickle(d, PLOT_DIR + 'slr3.pkl')


if __name__ == '__main__':
    main()
