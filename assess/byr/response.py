import numpy as np
import pandas as pd
from agent.util import load_valid_data, only_byr_agent
from assess.util import ll_wrapper, winzorize
from utils import topickle
from assess.const import BYR_NORM_DIMS
from constants import PLOT_DIR, IDX
from featnames import X_OFFER, INDEX, TEST, NORM, AUTO, CON, REJECT, EXP, BYR


def get_reindex_f(norm=None):
    return lambda s: s.reindex(index=norm.index, fill_value=False)


def get_feats(data=None, turn=None):
    assert turn in IDX[BYR][:-1]
    df0 = data[X_OFFER][[NORM, REJECT]].xs(turn, level=INDEX)
    df1 = data[X_OFFER][[AUTO, CON, REJECT, EXP]].xs(turn+1, level=INDEX)
    norm = df0[NORM]

    # running variable
    norm = norm[norm < 1]  # throw out accepts
    if turn > 1:  # throw out buyer rejections
        norm = norm[~df0[REJECT]]
    norm = winzorize(norm)
    f = get_reindex_f(norm)

    # create features
    autorej = f(df1[REJECT] & df1[AUTO])
    autoacc = f(~df1[REJECT] & df1[AUTO])
    acc = f((df1[CON] == 1) & ~df1[AUTO])
    rej = f(df1[REJECT] & ~df1[AUTO])
    counter = f((df1[CON] > 0) & df1[CON] < 1)
    exp = f(df1[EXP])
    end = pd.Series(False, index=df1.index).reindex(
        index=norm.index, fill_value=True)

    # put in dictionary
    y = {'Auto-accept': autoacc,
         'Manual accept': acc,
         'Counter': counter,
         'Auto-reject': autorej,
         'Manual reject': rej,
         'Expiration reject': exp,
         'Listing ends': end}

    for k, v in y.items():
        assert np.all(norm.index == v.index)
        y[k] = v.values

    return norm.values, y


def main():
    d = dict()

    # first threads in data
    data = only_byr_agent(load_valid_data(part=TEST, byr=True),
                          drop_thread=True)

    # response type ~ buyer offer
    for t in IDX[BYR][:-1]:
        key = 'area_response_{}'.format(t)
        x, y = get_feats(data=data, turn=t)
        flag = False
        for k, v in y.items():
            line, bw = ll_wrapper(x=x, y=v,
                                  dim=BYR_NORM_DIMS[t],
                                  ci=False,
                                  bw=(.05,))
            line = line.rename(k)

            if not flag:
                d[key] = line
                flag = True
            else:
                d[key] = pd.concat([d[key], line], axis=1)

        # normalize
        totals = d[key].sum(axis=1)
        for c in d[key].columns:
            d[key][c] /= totals

    topickle(d, PLOT_DIR + 'byrresponse.pkl')


if __name__ == '__main__':
    main()
