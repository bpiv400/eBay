import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import find_best_run, get_byr_agent, load_values
from assess.util import get_dim, kreg2
from utils import load_data, topickle, safe_reindex
from agent.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import X_OFFER, CON, NORM, INDEX, LSTG, EXP, AUTO, BYR, TEST


def main():
    d = {}
    for delta in DELTA_SLR:
        # byr run
        run_dir = find_best_run(byr=True, delta=delta)
        if run_dir is None:
            continue
        data = load_data(part=TEST, run_dir=run_dir)
        vals = load_values(part=TEST, delta=delta)

        # restrict to agent threads
        offers = safe_reindex(data[X_OFFER], idx=get_byr_agent(data))

        con = offers[CON].xs(3, level=INDEX)
        last = offers.xs(2, level=INDEX).reindex(index=con.index)
        norm = 1 - last[NORM]
        norm[~last[EXP] & ~last[AUTO]] = 1.05
        norm[last[EXP]] = 1.1
        y = con.values.astype(np.float64)
        x1 = norm.values
        x2 = vals.reindex(index=norm.index, level=LSTG).values

        # univariate regressions for rejections
        cols = {'Automatic': 1, 'Manual': 1.05, 'Expiration': 1.1}
        df = pd.DataFrame(columns=cols, index=get_dim(x2))
        for k, v in cols.items():
            rej = np.isclose(x1, v)
            ll = KernelReg(y[rej], x2[rej], var_type='c', bw=(.025,))
            df[k] = ll.fit(df.index)[0]
        d['simple_valcon_{}'.format(delta)] = df

        # bivariate kernel regression
        mask = x1 < 1
        d['contour_normval_{}'.format(delta)] = kreg2(y=y[mask],
                                                      x1=x1[mask],
                                                      x2=x2[mask])

    topickle(d, PLOT_DIR + '{}3.pkl'.format(BYR))


if __name__ == '__main__':
    main()
