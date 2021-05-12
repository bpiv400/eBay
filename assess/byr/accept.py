import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_sim_dir
from assess.util import ll_wrapper
from utils import topickle
from assess.const import POINTS
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX

DIM = np.linspace(.25, .75, POINTS)


def get_feats(data=None, t=None):
    slr_con = data[X_OFFER][CON].xs(t-1, level=INDEX)
    byr_acc = data[X_OFFER][CON].xs(t, level=INDEX) == 1
    slr_con = slr_con.loc[byr_acc.index]
    slr_con = slr_con[slr_con > 0]
    byr_acc = byr_acc.loc[slr_con.index]
    return byr_acc.values, slr_con.values


def main():
    d, bw = dict(), dict()
    prefix = 'response_conacc'

    data_obs = load_valid_data(byr=True, minimal=True)
    for t in [3, 5]:
        y, x = get_feats(data=data_obs, t=t)
        line, dots, bw[t] = ll_wrapper(y=y, x=x, discrete=[.5], dim=DIM)
        for obj in [line, dots]:
            obj.columns = pd.MultiIndex.from_product(
                [['Humans'], obj.columns])
        print('turn {}: {}'.format(t, bw[t][0]))
        d['{}_{}'.format(prefix, t)] = line, dots

    for delta in [1, 2]:
        sim_dir = get_sim_dir(byr=True, delta=delta)
        data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)
        for t in [3, 5]:
            y, x = get_feats(data=data_rl, t=t)
            line, dots, _ = ll_wrapper(y=y, x=x,
                                       dim=DIM,
                                       discrete=[.5],
                                       bw=bw[t],
                                       ci=False)
            cols = ('$\\lambda = {}$'.format(int(delta)), 'beta')
            d['{}_{}'.format(prefix, t)][0].loc[:, cols] = line
            d['{}_{}'.format(prefix, t)][1].loc[:, cols] = dots

    topickle(d, PLOT_DIR + 'byracc.pkl')


if __name__ == '__main__':
    main()
