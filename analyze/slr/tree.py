import argparse
import numpy as np
import pandas as pd
from sklearn import tree
from analyze.util import get_last
from agent.util import get_sim_dir, load_valid_data
from utils import safe_reindex
from agent.const import DELTA_SLR
from constants import DAY
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, LSTG, SIM, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, DAYS, TIME_FEATS


def estimate_tree(X=None, y=None, max_depth=1, criterion='entropy'):
    assert np.all(X.index == y.index)
    dt = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf = dt.fit(X.values, y.values)
    r = tree.export_text(clf, feature_names=list(X.columns))
    print(r)
    return r


def get_tree(data=None, turn=None, idx=None):
    # turn-specific dictionary
    valid = data[X_OFFER][~data[X_OFFER][AUTO]].xs(
        turn, level=INDEX, drop_level=False).index
    d = safe_reindex(data, idx=valid)
    for k, v in d.items():
        d[k] = v.droplevel(INDEX)

    # restrict to provided indices
    if idx is not None:
        assert turn > 2
        idx = idx.intersection(d[CLOCK].index)
        if len(idx) == 0:
            return None
        d = {k: safe_reindex(v, idx=idx) for k, v in d.items()}

    # outcome
    y = d[X_OFFER][CON].astype(str)
    u = np.unique(y)
    print(u)
    if len(u) == 1:
        print('Turn {}:'.format(turn + 2))
        get_tree(data=data, turn=turn+2)
        return None

    con_rate = y.groupby(y).count() / len(y)
    con_rate = con_rate[con_rate > .01]
    print(con_rate)

    # features
    X = pd.concat([d[LOOKUP][START_PRICE],
                   d[X_THREAD][BYR_HIST],
                   d[X_OFFER][TIME_FEATS[6:14:2]]], axis=1)
    X[DAYS] = (d[CLOCK] - d[LOOKUP][START_TIME]) / DAY
    X[NORM] = get_last(data[X_OFFER][NORM]).xs(turn, level=INDEX).loc[X.index]
    assert X.isna().sum().sum() == 0

    # decision tree
    r = estimate_tree(X=X, y=y)

    # find split
    if turn < 6:
        feat = r.split(' ')[1]
        val = float(r.split(' ')[3][:-2])
        idx1 = r.find('class:', 0, -1) + 7
        idx2 = r.find('class:', idx1, -1) + 7
        low = r[idx1:].split('\n')[0]
        high = r[idx2:].split('\n')[0]
        if low == high:  # no split
            print('Turn {}:'.format(turn+2))
            idx = y[y == low].index
            get_tree(data=data, turn=turn+2, idx=idx)
        else:
            print('Turn {}, {} <= {}:'.format(turn+2, feat, val))
            idx_l = y[(X[feat] <= val) & (y == low)].index
            get_tree(data=data, turn=turn+2, idx=idx_l)

            print('Turn {}, {} > {}:'.format(turn + 2, feat, val))
            idx_h = y[(X[feat] > val) & (y == high)].index
            get_tree(data=data, turn=turn+2, idx=idx_h)


def main():
    # agent params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_SLR)
    delta = parser.parse_args().delta

    # load data
    sim_dir = get_sim_dir(delta=delta)
    data = load_valid_data(sim_dir=sim_dir, clock=True)
    data[X_OFFER][TIME_FEATS] = data[X_OFFER][TIME_FEATS].groupby([LSTG, SIM]).cumsum()

    # start recursion
    print('Turn 2:')
    get_tree(data=data, turn=2)


if __name__ == '__main__':
    main()
