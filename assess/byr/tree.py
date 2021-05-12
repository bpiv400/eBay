import argparse
import numpy as np
import pandas as pd
from agent.util import get_sim_dir, load_valid_data
from assess.util import estimate_tree
from utils import load_feats, safe_reindex
from agent.const import DELTA_BYR
from constants import DAY, MAX_DAYS
from featnames import LOOKUP, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, REJECT, MSG, AUTO

LISTING_FEATS = ['fdbk_score', 'fdbk_pstv', 'photos', 'store', 'slr_us', 'fast']


def get_tree(data=None, turn=None, idx=None):
    if turn == 7:
        return

    # turn-specific dictionary
    d = {k: data[k].xs(turn, level=INDEX) for k in [X_OFFER, CLOCK]}
    for k in ['listings', LOOKUP, X_THREAD]:
        d[k] = safe_reindex(data[k], idx=d[CLOCK].index)

    # restrict to provided indices
    if idx is not None:
        assert turn > 1
        idx = idx.intersection(d[CLOCK].index)
        if len(idx) == 0:
            return
        d = {k: safe_reindex(v, idx=idx) for k, v in d.items()}

    # outcome
    y = d[X_OFFER][CON].astype(str)
    u = np.unique(y)
    print(u)
    if len(u) == 1:
        print('Turn {}:'.format(turn + 2))
        get_tree(data=data, turn=turn+2)
        return

    con_rate = y.groupby(y).count() / len(y)
    con_rate = con_rate[con_rate > .01]
    print(con_rate)

    # features
    X = pd.concat([d['listings'], d[LOOKUP][START_PRICE], d[X_THREAD][BYR_HIST]],
                  axis=1)
    X['elapsed'] = (d[CLOCK] - d[LOOKUP][START_TIME]) / (MAX_DAYS * DAY)
    if turn > 1:
        last = data[X_OFFER][[NORM, REJECT, AUTO, MSG]].xs(
            turn - 1, level=INDEX).loc[d[CLOCK].index]
        last[NORM] = 1 - last[NORM]
        X = X.join(last)

    # decision tree
    r = estimate_tree(X=X, y=y)

    # find split
    feat = r.split(' ')[1]
    val = float(r.split(' ')[3][:-2])
    idx1 = r.find('class:', 0, -1) + 7
    idx2 = r.find('class:', idx1, -1) + 7
    low = r[idx1:].split('\n')[0]
    high = r[idx2:].split('\n')[0]

    if low == high:  # no split
        print('Turn {}:'.format(turn+2))
        mask = y == low
        idx = mask[mask].index
        get_tree(data=data, turn=turn+2, idx=idx)
    else:
        print('Turn {}, {} <= {}:'.format(turn+2, feat, val))
        mask_l = (X[feat] <= val) & (y == low)
        idx_l = mask_l[mask_l].index
        get_tree(data=data, turn=turn+2, idx=idx_l)

        print('Turn {}, {} > {}:'.format(turn + 2, feat, val))
        mask_h = (X[feat] > val) & (y == high)
        idx_h = mask_h[mask_h].index
        get_tree(data=data, turn=turn+2, idx=idx_h)


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_BYR)
    delta = parser.parse_args().delta

    sim_dir = get_sim_dir(byr=True, delta=delta)
    data = load_valid_data(sim_dir=sim_dir, clock=True, minimal=True)

    # add listing features to data
    listings = load_feats('listings')[LISTING_FEATS]
    data['listings'] = safe_reindex(listings, idx=data[LOOKUP].index)

    # start recursion
    print('Turn 1:')
    get_tree(data=data, turn=1)


if __name__ == '__main__':
    main()
