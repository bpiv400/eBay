import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_run_dir
from utils import topickle, safe_reindex
from agent.const import DELTA_SLR
from constants import PLOT_DIR, IDX, SLR
from featnames import START_PRICE, LOOKUP, X_OFFER, X_THREAD, NORM, DAYS_SINCE_LSTG


def get_con(data=None):
    norm = data[X_OFFER][NORM].unstack()
    norm.loc[:, IDX[SLR]] = 1 - norm.loc[:, IDX[SLR]]
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=norm.index)
    for t in norm.columns:
        norm[t] *= start_price
    con = pd.DataFrame(index=norm.index)
    con[1] = norm[1]
    con[2] = start_price - norm[2]
    for t in range(3, 8):
        con[t] = np.abs(norm[t - 2] - norm[t])
    for t in con.columns:
        con[t] /= start_price
    return con.fillna(0)


def get_day(data=None):
    return 1 + np.floor(data[X_THREAD][DAYS_SINCE_LSTG])


def get_df(data=None):
    con = get_con(data=data)
    day = get_day(data=data)
    assert np.all(con.index == day.index)
    df = pd.concat([con[t].groupby(day).mean() for t in con.columns], axis=1)
    return df


def main():
    d, prefix = dict(), 'area_turncon'

    # human sellers
    data_obs = load_valid_data(byr=False, minimal=True)
    d['{}_Humans'.format(prefix)] = get_df(data=data_obs)

    # agent sellers
    for delta in DELTA_SLR:
        run_dir = get_run_dir(delta=delta)
        data_rl = load_valid_data(run_dir=run_dir, minimal=True)
        d['{}_{}'.format(prefix, delta)] = get_df(data=data_rl)

    topickle(d, PLOT_DIR + 'slrcon.pkl')


if __name__ == '__main__':
    main()
