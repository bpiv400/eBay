import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_sim_dir
from analyze.util import save_dict
from featnames import X_OFFER, NORM, SLR, LOOKUP, START_PRICE
from utils import safe_reindex
from agent.const import DELTA_BYR
from constants import IDX


def get_total_con(data=None):
    norm = data[X_OFFER][NORM].unstack()
    norm = norm[norm[1] < 1]
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
    return con.fillna(0).mean()


def main():
    df = pd.DataFrame()

    # human sellers
    data_obs = load_valid_data(byr=True, minimal=True)
    df['Humans'] = get_total_con(data=data_obs)

    # agent buyers
    for delta in DELTA_BYR:
        run_dir = get_sim_dir(byr=True, delta=delta)
        data_rl = load_valid_data(sim_dir=run_dir, minimal=True)
        df['$\\lambda = {}$'.format(delta)] = get_total_con(data=data_rl)

    d = {'stacked_con': df}
    save_dict(d, 'byrcon')


if __name__ == '__main__':
    main()
