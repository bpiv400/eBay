import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_sim_dir
from assess.util import get_total_con
from utils import topickle
from agent.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import X_THREAD, DAYS_SINCE_LSTG


def get_df(data=None):
    con = get_total_con(data=data)
    day = 1 + np.floor(data[X_THREAD][DAYS_SINCE_LSTG])
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
        run_dir = get_sim_dir(delta=delta)
        data_rl = load_valid_data(sim_dir=run_dir, minimal=True)
        d['{}_{}'.format(prefix, delta)] = get_df(data=data_rl)

    topickle(d, PLOT_DIR + 'slrcon.pkl')


if __name__ == '__main__':
    main()
