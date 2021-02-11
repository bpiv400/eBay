import pandas as pd
from agent.util import load_valid_data, load_values, get_run_dir
from assess.util import bin_vs_reward, get_eval_df
from utils import topickle, safe_reindex
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import PLOT_DIR
from featnames import START_PRICE, LOOKUP, X_OFFER, NORM, AUTO


def mean_con(data=None):
    auto = data[X_OFFER][AUTO]
    norm = 1 - data[X_OFFER].loc[~auto, NORM].unstack()[[2, 4, 6]]
    norm = norm[norm.count(axis=1) > 0]
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=norm.index)
    for t in norm.columns:
        norm[t] *= start_price

    con = pd.DataFrame(index=norm.index)
    con[2] = start_price - norm[2]

    for t in [4, 6]:
        con[t] = norm[t - 2] - norm[t]
        mask = norm[t - 2].isna() & ~norm[t].isna()
        con.loc[mask, t] = start_price.loc[mask] - norm.loc[mask, t]

    return con.mean()


def main():
    # human sellers
    data_obs = load_valid_data(byr=False, minimal=True)
    con_obs = mean_con(data_obs)
    print(con_obs)

    for delta in DELTA_SLR:
        run_dir = get_run_dir(delta=delta)
        data_rl = load_valid_data(run_dir=run_dir, minimal=True)
        con_rl = mean_con(data_rl)
        print(con_rl)


if __name__ == '__main__':
    main()
