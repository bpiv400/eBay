from copy import deepcopy
import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_output_dir
from assess.util import ll_wrapper
from utils import topickle, safe_reindex
from agent.const import TURN_COST_CHOICES, DELTA_BYR
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import LOOKUP, START_PRICE


def get_x(data=None, idx=None):
    return safe_reindex(np.log10(data[LOOKUP][START_PRICE]), idx=idx)


def get_feats(data=None, get_y=None):
    y = get_y(data=data)
    x = get_x(data=data, idx=y.index)
    return y.values, x.values


def bin_plot(name=None, get_y=None):
    d, means = dict(), pd.Series(name=name, index=['Humans'] + DELTA_BYR)

    # humans
    data_obs = only_byr_agent(load_valid_data(byr=True, minimal=True))
    y, x = get_feats(data=data_obs, get_y=get_y)
    means.loc['Humans'] = y.mean()

    # by list price
    line, bw = ll_wrapper(y=y, x=x, dim=LOG10_BIN_DIM)
    line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
    print('bw: {}'.format(bw[0]))

    # turn cost comparison
    key = 'simple_list{}'.format(name)
    d[key] = deepcopy(line)
    for t in TURN_COST_CHOICES:
        run_dir = get_output_dir(byr=True, delta=1, turn_cost=t)
        data_rl = only_byr_agent(load_valid_data(run_dir=run_dir,
                                                 minimal=True))
        y, x = get_feats(data=data_rl, get_y=get_y)
        line, _ = ll_wrapper(y=y, x=x,
                             dim=LOG10_BIN_DIM,
                             bw=bw,
                             ci=False)
        line.loc[line.index < np.quantile(x, .05)] = np.nan
        d[key].loc[:, ('${}'.format(t), 'beta')] = line
        if t == 0:
            means.loc[1.] = y.mean()

    # bar chart of means
    for delta in DELTA_BYR:
        if np.isnan(means.loc[delta]):
            run_dir = get_output_dir(byr=True, delta=delta)
            data_rl = only_byr_agent(load_valid_data(run_dir=run_dir,
                                                     minimal=True))
            means.loc[delta] = get_y(data=data_rl).mean()

    d['bar_list{}'.format(name)] = means

    topickle(d, PLOT_DIR + 'byr{}.pkl'.format(name))
