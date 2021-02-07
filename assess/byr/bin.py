import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import ll_wrapper
from utils import topickle, safe_reindex
from agent.const import TURN_COST_CHOICES
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, LOOKUP, START_PRICE


def get_feats(data=None):
    con1 = data[X_OFFER][CON].xs(1, level=INDEX)
    assert not any(con1 == 0)
    acc1 = con1 == 1
    log10bin = safe_reindex(np.log10(data[LOOKUP][START_PRICE]),
                            idx=acc1.index)
    return acc1.values, log10bin.values


def main():
    d = dict()

    # humans
    data_obs = only_byr_agent(load_valid_data(byr=True,
                                              minimal=True))

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

    d['simple_binacc'] = line

    topickle(d, PLOT_DIR + 'byrbin.pkl')


if __name__ == '__main__':
    main()
