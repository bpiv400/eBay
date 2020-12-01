import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import find_best_run, get_byr_agent, load_values
from assess.util import kreg2
from utils import load_data, topickle
from agent.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import X_OFFER, CON, NORM, INDEX, LSTG, EXP, BYR, TEST


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
        idx = get_byr_agent(data)
        offers = pd.DataFrame(index=idx).join(data[X_OFFER])

        con = offers[CON].xs(5, level=INDEX)
        last = offers.xs(4, level=INDEX).reindex(index=con.index)
        norm = 1 - last[NORM]
        # norm[(last[CON] == 0) & (last[DELAY] == 1)] = 1.1
        y = con.values.astype(np.float64)
        x1 = norm.values
        x2 = vals.reindex(index=norm.index, level=LSTG).values

        # # univariate regressions for rejections
        # cols = {'Non-expiration': 1, 'Expiration': 1.1}
        # df = pd.DataFrame(columns=cols, index=get_dim(x2))
        # for k, v in cols.items():
        #     ll = KernelReg(y, x2, var_type='c', bw=(.025,))
        #     df[k] = ll.fit(df.index)[0]
        #     print('{}: {}'.format(k, ll.bw[0]))
        # d['simple_valcon_{}'.format(delta)] = df

        # bivariate kernel regression
        mask = x1 < 1
        d['contour_normval_{}'.format(delta)] = kreg2(y=y[mask],
                                                      x1=x1[mask],
                                                      x2=x2[mask])

    topickle(d, PLOT_DIR + '{}3.pkl'.format(BYR))


if __name__ == '__main__':
    main()
