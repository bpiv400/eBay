import pandas as pd
from agent.eval.util import get_eval_path
from agent.util import load_valid_data, load_values, get_sim_dir
from assess.util import bin_vs_reward
from utils import topickle, unpickle
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import PLOT_DIR


def main():
    d = dict()

    # human sellers
    data_obs = load_valid_data(byr=False)

    # evals
    evals = unpickle(get_eval_path(byr=False))

    # seller
    for delta in DELTA_SLR:
        # average reward at each list price
        key = 'simple_rewardbin_{}'.format(delta)
        values = delta * load_values(delta=delta)

        line, bw = bin_vs_reward(data=data_obs, values=values)
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d[key] = line
        print('{}: {}'.format(delta, bw[0]))

        sim_dir = get_sim_dir(delta=delta)
        data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)

        line, _ = bin_vs_reward(data=data_rl, values=values, bw=bw)
        d[key].loc[:, (SLR_NAMES[delta], 'beta')] = line

        # bar chart of reward
        for c in ['norm', 'dollar']:
            d['bar_{}_{}'.format(c, delta)] = evals[SLR_NAMES[delta]].loc[c, :]

    topickle(d, PLOT_DIR + 'slreval.pkl')


if __name__ == '__main__':
    main()
