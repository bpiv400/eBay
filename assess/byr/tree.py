import argparse
import numpy as np
import pandas as pd
from agent.util import get_sim_dir, only_byr_agent, load_valid_data
from assess.util import estimate_tree
from utils import load_feats, safe_reindex
from agent.const import DELTA_BYR
from constants import DAY, MAX_DAYS, IDX
from featnames import LOOKUP, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, BYR, REJECT, MSG, AUTO

LISTING_FEATS = ['fdbk_score', 'fdbk_pstv', 'photos', 'store', 'slr_us', 'fast']


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_BYR)
    delta = parser.parse_args().delta

    sim_dir = get_sim_dir(byr=True, delta=delta)
    data = only_byr_agent(load_valid_data(sim_dir=sim_dir, clock=True, minimal=True))

    turns = IDX[BYR] if delta < 1 else IDX[BYR][:-1]
    for turn in turns:
        print('Turn {}'.format(turn))

        d = {k: data[k].xs(turn, level=INDEX) for k in [X_OFFER, CLOCK]}
        for k in [LOOKUP, X_THREAD]:
            d[k] = safe_reindex(data[k], idx=d[CLOCK].index)

        # outcome
        con = d[X_OFFER][CON]
        y = pd.Series('', index=con.index)
        y.loc[con == 0] = 'Walk'
        y.loc[con == 1] = 'Accept'
        y.loc[np.isclose(con, .5)] = 'Half'
        y.loc[(con > 0) & (con < .5)] = 'Low'
        y.loc[(con > .5) & (con < 1)] = 'High'
        print(np.unique(y))

        con_rate = con.groupby(con).count() / len(con)
        con_rate = con_rate[con_rate > .01]
        print(con_rate)

        # features
        X = d[X_THREAD][BYR_HIST].to_frame().join(d[LOOKUP][START_PRICE])
        X['elapsed'] = (d[CLOCK] - d[LOOKUP][START_TIME]) / (MAX_DAYS * DAY)
        if turn > 1:
            last = 1 - data[X_OFFER][NORM].xs(turn-1, level=INDEX).loc[d[CLOCK].index]
            X = X.join(last)

        # decision tree
        estimate_tree(X=X, y=y)


if __name__ == '__main__':
    main()
