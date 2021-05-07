import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_sim_dir, only_byr_agent
from assess.util import get_total_con
from utils import topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR


def get_column(data=None):
    con = get_total_con(data=data)
    sales = np.isclose(con.sum(axis=1), 1)
    return con.loc[sales].mean()


def main():
    df = pd.DataFrame()

    # human sellers
    data_obs = only_byr_agent(load_valid_data(byr=False, minimal=True))
    df['Humans'] = get_column(data=data_obs)

    # agent buyers
    for delta in DELTA_BYR:
        run_dir = get_sim_dir(byr=True, delta=delta)
        data_rl = only_byr_agent(load_valid_data(sim_dir=run_dir, minimal=True))
        df['$\\lambda = {}$'.format(delta)] = get_column(data=data_rl)

    d = {'stacked_con': df}
    topickle(d, PLOT_DIR + 'byrcon.pkl')


if __name__ == '__main__':
    main()
