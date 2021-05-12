import pandas as pd
from agent.util import load_valid_data, get_run_dir
from utils import topickle
from constants import PLOT_DIR
from featnames import X_OFFER, NORM, CON, LOOKUP, START_PRICE


def get_path(data=None):
    con = data[X_OFFER][CON].unstack()
    norm = data[X_OFFER][NORM].unstack()
    norm = norm.loc[(con[5] > 0) & (con[5] < 1), range(1, 7)]
    norm.loc[norm[6].isna(), 6] = 0
    start_price = data[LOOKUP][START_PRICE].loc[norm.index]
    for t in norm.columns:
        if t in [2, 4, 6]:
            norm[t] = 1 - norm[t]
        norm[t] *= start_price
    norm[0] = start_price
    norm = norm.sort_index(axis=1)
    print(norm.mean())
    return norm.mean()


def main():
    df = pd.DataFrame(index=range(1, 8))

    data_obs = load_valid_data(byr=True, minimal=True)
    df['Humans'] = get_path(data_obs)

    for delta in [.9, 1, 2]:
        run_dir = get_run_dir(byr=True, delta=delta)
        data_rl = load_valid_data(sim_dir=run_dir, minimal=True)
        df['$\\lambda = {}$'.format(delta)] = get_path(data_rl)

    d = {'simple_conpath': df}

    topickle(d, PLOT_DIR + 'byrpath.pkl')


if __name__ == '__main__':
    main()
