import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_sim_dir
from assess.util import ll_wrapper
from utils import topickle
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, NORM, AUTO, EXP

DIM = np.linspace(.7, 1, 100)


def get_feats(data=None, turn=None):
    assert turn in [3, 5]
    # whether buyer counters in turn
    byrcon = data[X_OFFER][CON].xs(turn, level=INDEX)
    byrcounter = (byrcon > 0) & (byrcon < 1)
    # seller offer in turn - 1
    df0 = data[X_OFFER].xs(turn-1, level=INDEX)
    active = ~df0[AUTO] & ~df0[EXP]
    slrnorm = (1 - df0[NORM])[active]
    # common index
    idx = byrcon.index.intersection(slrnorm.index)
    # features
    y = byrcounter.loc[idx].values
    x = slrnorm.loc[idx].values
    return y, x


def main():
    d, bw, prefix = dict(), dict(), 'response_counter'

    data_obs = load_valid_data(byr=True, minimal=True)

    for t in [3, 5]:
        y, x = get_feats(data=data_obs, turn=t)
        line, dots, bw[t] = ll_wrapper(y=y, x=x, dim=DIM, discrete=[1.])
        print('bw for turn {}: {}'.format(t, bw[t][0]))
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        dots.columns = pd.MultiIndex.from_product([['Humans'], dots.columns])
        d['{}_{}'.format(prefix, t)] = line, dots

    for delta in [1, 1.5, 2]:
        sim_dir = get_sim_dir(byr=True, delta=delta)
        data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)

        for t in [3, 5]:
            y, x = get_feats(data=data_rl, turn=t)
            line, dots, _ = ll_wrapper(y=y, x=x, dim=DIM,
                                       discrete=[1.], bw=bw[t], ci=False)
            key = '{}_{}'.format(prefix, t)
            cols = ('$\\lambda = {}$'.format(delta), 'beta')
            d[key][0].loc[:, cols] = line
            d[key][1].loc[:, cols] = dots

    topickle(d, PLOT_DIR + 'byrresponse.pkl')


if __name__ == '__main__':
    main()
