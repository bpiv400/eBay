import argparse
import numpy as np
import pandas as pd
from agent.util import get_run_dir, get_byr_agent
from assess.util import estimate_tree
from utils import load_data, load_feats, safe_reindex, compose_args
from agent.const import AGENT_PARAMS
from constants import DAY, MAX_DAYS, IDX
from featnames import LOOKUP, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, THREAD, TEST, BYR, AUTO, EXP, MSG

LISTING_FEATS = ['fdbk_score', 'fdbk_pstv', 'photos', 'store', 'slr_us', 'fast']


def main():
    # agent params from command line
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    params = vars(parser.parse_args())
    assert params[BYR]

    run_dir = get_run_dir(**params)
    data = load_data(part=TEST, run_dir=run_dir)

    # add listing features to data
    listings = load_feats('listings')[LISTING_FEATS]
    data['listings'] = safe_reindex(listings, idx=data[LOOKUP].index)

    # restrict to byr agent
    threads = get_byr_agent(data)
    for k, v in data.items():
        data[k] = safe_reindex(v, idx=threads).droplevel(THREAD)

    for turn in IDX[BYR]:
        print('Turn {}'.format(turn))

        d = {k: data[k].xs(turn, level=INDEX) for k in [X_OFFER, CLOCK]}
        for k in ['listings', LOOKUP, X_THREAD]:
            d[k] = safe_reindex(data[k], idx=d[CLOCK].index)

        # outcome
        con = d[X_OFFER][CON]
        y = pd.Series('', index=con.index)
        y.loc[con == 0] = 'Walk'
        y.loc[con == 1] = 'Accept'
        if turn == 1:
            y.loc[(con > 0) & (con <= .5)] = 'Low'
            y.loc[(con < .5) & (con > 1)] = 'High'
        else:
            y.loc[(con > 0) & (con <= .25)] = 'Low'
            y.loc[(con < 1) & (con > .25)] = 'High'
        print(np.unique(y))

        con_rate = con.groupby(con).count() / len(con)
        con_rate = con_rate[con_rate > .01]
        print(con_rate)

        # features
        X = d[X_THREAD][BYR_HIST].to_frame()
        X['elapsed'] = (d[CLOCK] - d[LOOKUP][START_TIME]) / (MAX_DAYS * DAY)
        if turn > 1:
            last = data[X_OFFER][[CON, NORM, AUTO, EXP, MSG]].xs(
                turn-1, level=INDEX).loc[d[CLOCK].index]
            last[NORM] = 1 - last[NORM]
            X = X.join(last)
        X = X.join(d[LOOKUP][START_PRICE]).join(d['listings'])

        # decision tree
        estimate_tree(X=X, y=y)


if __name__ == '__main__':
    main()
