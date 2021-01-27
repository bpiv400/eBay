import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import continuous_cdf, ll_wrapper
from utils import topickle, safe_reindex
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, NORM, TEST, LOOKUP, START_PRICE, THREAD


def main():
    data = only_byr_agent(load_valid_data(part=TEST, byr=True))

    # turn 2
    norm = data[X_OFFER][NORM].unstack().droplevel(THREAD)
    norm1 = norm[1][norm[1] < 1]
    discount2 = norm[2].loc[norm1.index]
    idx = discount2[~discount2.isna()].index

    y = discount2.loc[idx].values
    x = np.log10(data[LOOKUP][START_PRICE].loc[idx].values)

    line, bw = ll_wrapper(y=y, x=x, dim=LOG10_BIN_DIM)


if __name__ == '__main__':
    main()
