import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import ll_wrapper
from utils import topickle, safe_reindex
from agent.const import TURN_COST_CHOICES
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, LOOKUP, START_PRICE


def get_feats(data=None):
    con = data[X_OFFER][CON].unstack()[[1, 3, 5]]
    count = ((0 < con) & (con < 1)).sum(axis=1)
    count = count[count > 0]
    log10bin = safe_reindex(np.log10(data[LOOKUP][START_PRICE]),
                            idx=count.index)
    return count.values, log10bin.values


def main():
    d = dict()

    # humans
    data_obs = only_byr_agent(load_valid_data(byr=True, minimal=True))

    y, x = get_feats(data=data_obs)
    line, bw = ll_wrapper(y=y, x=x, dim=LOG10_BIN_DIM)
    line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
    print('bw: {}'.format(bw[0]))

    for t in TURN_COST_CHOICES:
        run_dir = get_run_dir(byr=True, delta=1, turn_cost=t)
        data_rl = only_byr_agent(load_valid_data(run_dir=run_dir,
                                                 minimal=True))
        if data_rl is not None:
            y, x = get_feats(data=data_rl)
            line.loc[:, ('{}'.format(t), 'beta')], _ = \
                ll_wrapper(y=y, x=x, dim=LOG10_BIN_DIM, bw=bw, ci=False)

    d['simple_binoffers'] = line

    topickle(d, PLOT_DIR + 'byroffers.pkl')


if __name__ == '__main__':
    main()
