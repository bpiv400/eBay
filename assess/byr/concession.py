import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import ll_wrapper
from utils import topickle
from assess.const import POINTS
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX


def main():
    run_dir = get_run_dir(byr=True, delta=2)
    data_rl = only_byr_agent(load_valid_data(run_dir=run_dir,
                                             minimal=True))

    con = data_rl[X_OFFER][CON].unstack()
    for t in [3, 5]:
        s = con[t].dropna()
        s = s[(s > 0) & (s < 1)]
        pdf = s.groupby(s).count() / len(s)
        print(pdf)


if __name__ == '__main__':
    main()
