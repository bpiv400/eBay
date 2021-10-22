import numpy as np
import pandas as pd
from agent.util import load_valid_data, load_values, get_sim_dir, get_norm_reward
from analyze.util import ll_wrapper, save_dict
from agent.const import DELTA_SLR
from analyze.const import SLR_NAMES, LOG10_BIN_DIM
from featnames import LOOKUP, START_PRICE

def bin_vs_reward(data=None, values=None, bw=None):
    reward = pd.concat(get_norm_reward(data=data, values=values, byr=False))
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
    data_obs = load_valid_data(byr=False, minimal=True)

    # seller
    for delta in DELTA_SLR:
        # average reward at each list price
        key = 'simple_rewardbin_{}'.format(delta)
        values = delta * load_values(delta=delta)

        line, bw = bin_vs_reward(data=data_obs, values=values)
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d[key] = line
        print('{}: {}'.format(delta, bw[0]))

        sim_dir = get_sim_dir(byr=False, delta=delta)
        data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)

        line, _ = bin_vs_reward(data=data_rl, values=values, bw=bw)
        name = SLR_NAMES[delta]
        d[key].loc[:, (name, 'beta')] = line

        h_dir = get_sim_dir(byr=False, delta=delta, heuristic=True)
        data_h = load_valid_data(sim_dir=h_dir, minimal=True)

        line, _ = bin_vs_reward(data=data_h, values=values, bw=bw)
        name = 'Heuristic {}'.format(name.lower())
        d[key].loc[:, (name, 'beta')] = line

    # save
    save_dict(d, 'slreval')


if __name__ == '__main__':
    main()
