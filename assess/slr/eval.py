import numpy as np
import pandas as pd
from agent.util import load_valid_data, load_values, get_norm_reward, \
    get_run_dir
from assess.util import get_eval_df, ll_wrapper
from utils import topickle
from agent.const import DELTA_SLR
from assess.const import LOG10_BIN_DIM, SLR_NAMES
from constants import PLOT_DIR
from featnames import SLR, TEST, LOOKUP, START_PRICE


def bin_vs_reward(data=None, values=None, bw=None):
    reward = pd.concat(get_norm_reward(data=data, values=values))
    y = reward.reindex(index=data[LOOKUP].index).values
    x = np.log10(data[LOOKUP][START_PRICE].values)
    line, bw = ll_wrapper(y=y,
                          x=x,
                          dim=LOG10_BIN_DIM,
                          bw=bw,
                          ci=(bw is None))
    return line, bw


def main():
    d = dict()

    # human sellers
    data_obs = load_valid_data(part=TEST, byr=False)

    # seller
    for delta in DELTA_SLR:
        # average reward at each list price
        key = 'simple_rewardbin_{}'.format(delta)
        values = delta * load_values(part=TEST, delta=delta)

        line, bw = bin_vs_reward(data=data_obs, values=values)
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d[key] = line
        print('{}: {}'.format(delta, bw[0]))

        run_dir = get_run_dir(byr=False, delta=delta)
        data_rl = load_valid_data(part=TEST, run_dir=run_dir)
        line, _ = bin_vs_reward(data=data_rl, values=values, bw=bw)
        d[key].loc[:, (SLR_NAMES[delta], 'beta')] = line

        # bar chart of reward
        df = get_eval_df(byr=False, delta=delta)
        for c in df.columns:
            d['bar_{}_{}'.format(c, delta)] = df[c]

    topickle(d, PLOT_DIR + '{}eval.pkl'.format(SLR))


if __name__ == '__main__':
    main()
