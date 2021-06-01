import pandas as pd
from agent.util import load_valid_data, get_sim_dir
from assess.util import get_total_con
from utils import topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR


def main():
    df = pd.DataFrame()

    # human sellers
    data_obs = load_valid_data(byr=True, minimal=True)
    df['Humans'] = get_total_con(data=data_obs, drop_bin=True).mean()

    # agent buyers
    for delta in DELTA_BYR:
        run_dir = get_sim_dir(byr=True, delta=delta)
        data_rl = load_valid_data(sim_dir=run_dir, minimal=True)
        df['$\\lambda = {}$'.format(delta)] = \
            get_total_con(data=data_rl, drop_bin=True).mean()

    d = {'stacked_con': df}
    topickle(d, PLOT_DIR + 'byrcon.pkl')


if __name__ == '__main__':
    main()
