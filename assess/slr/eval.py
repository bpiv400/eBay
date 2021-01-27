import pandas as pd
from agent.util import load_valid_data, load_values, get_run_dir
from assess.util import get_eval_df, bin_vs_reward
from utils import topickle
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import PLOT_DIR
from featnames import TEST


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

        run_dir = get_run_dir(delta=delta)
        data_rl = load_valid_data(part=TEST, run_dir=run_dir)

        line, _ = bin_vs_reward(data=data_rl, values=values, bw=bw)
        d[key].loc[:, (SLR_NAMES[delta], 'beta')] = line

        # bar chart of reward
        df = get_eval_df(delta=delta)
        for c in df.columns:
            d['bar_{}_{}'.format(c, delta)] = df[c]

    topickle(d, PLOT_DIR + 'slreval.pkl')


if __name__ == '__main__':
    main()
