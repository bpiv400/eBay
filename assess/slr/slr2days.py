import numpy as np
from agent.util import get_run_dir, load_valid_data, get_slr_valid
from assess.util import kreg2
from utils import topickle, load_data, load_feats, safe_reindex
from agent.const import DELTA_SLR
from assess.const import NORM1_DIM_SHORT
from constants import PLOT_DIR
from featnames import X_OFFER, REJECT, CON, INDEX, TEST, NORM, AUTO, EXP, \
    X_THREAD, THREAD, DAYS_SINCE_LSTG, STORE, LOOKUP


def get_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO] & ~data[X_OFFER][EXP],
                           [CON, REJECT, NORM]].xs(1, level=THREAD)
    acc2 = df[CON].xs(2, level=INDEX) == 1
    rej2 = df[REJECT].xs(2, level=INDEX)
    norm1 = df[NORM].xs(1, level=INDEX).loc[rej2.index]
    days = data[X_THREAD][DAYS_SINCE_LSTG].xs(1, level=THREAD).loc[rej2.index]
    return acc2.values, rej2.values, norm1.values, days.values


def main():
    d, bw = dict(), dict()

    # output mesh
    xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, np.linspace(.5, 6, 50))
    mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

    # observed sellers, by store
    data = get_slr_valid(load_data(part=TEST))
    store = load_feats('listings', lstgs=data[LOOKUP].index)[STORE]
    obs = {'nostore': {k: safe_reindex(v, idx=store[~store].index)
                       for k, v in data.items()},
           'store': {k: safe_reindex(v, idx=store[store].index)
                     for k, v in data.items()}}

    for k in ['store', 'nostore']:
        y_acc, y_rej, x1, x2 = get_feats(data=obs[k])
        for feat in ['acc', 'rej']:
            y = locals()['y_{}'.format(feat)]
            key = 'contour_{}days_{}'.format(feat, k)
            if feat not in bw:
                d[key], bw[feat] = kreg2(y=y, x1=x1, x2=x2, mesh=mesh)
                print('{}: {}'.format(feat, bw[feat]))
            else:
                d[key], _ = kreg2(y=y, x1=x1, x2=x2, mesh=mesh, bw=bw[feat])

    # seller runs
    for delta in DELTA_SLR:
        run_dir = get_run_dir(delta=delta)
        data = load_valid_data(part=TEST, run_dir=run_dir)
        y_acc, y_rej, x1, x2 = get_feats(data=data)
        for feat in ['acc', 'rej']:
            y = locals()['y_{}'.format(feat)]
            d['contour_{}days_{}'.format(feat, delta)], _ = \
                kreg2(y=y, x1=x1, x2=x2, mesh=mesh, bw=bw[feat])

    topickle(d, PLOT_DIR + 'slr2days.pkl')


if __name__ == '__main__':
    main()
