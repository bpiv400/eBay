import numpy as np
import pandas as pd
from agent.util import load_valid_data, only_byr_agent
from assess.util import ll_wrapper
from utils import topickle
from assess.const import BYR_NORM_DIMS
from constants import PLOT_DIR, HOUR, IDX
from featnames import X_OFFER, INDEX, TEST, NORM, CLOCK, AUTO, REJECT, BYR


def get_feats(data=None, turn=None):
    assert turn in IDX[BYR][:-1]
    df = data[X_OFFER][[NORM, AUTO, REJECT]]
    norm = df[NORM].xs(turn, level=INDEX)
    norm = norm[norm < 1]  # throw out accepts
    if turn > 1:  # throw out buyer rejections
        norm = norm[~df[REJECT].xs(turn, level=INDEX)]

    # create features
    wide = data[CLOCK].unstack().loc[norm.index]
    tdiff = wide[turn + 1] - wide[turn]
    hours = tdiff[~tdiff.isna()] / HOUR
    manhours = hours[hours > 0]

    # put in dictionary
    feats = {'Excl. auto': (norm.loc[manhours.index], manhours),
             'Incl. auto': (norm.loc[hours.index], hours)}

    for k, v in feats.items():
        assert np.all(v[0].index == v[1].index)
        feats[k] = (v[0].values, v[1].values)

    return feats


def main():
    d = dict()

    # first threads in data
    data = only_byr_agent(load_valid_data(part=TEST, byr=True),
                          drop_thread=True)

    # response type ~ buyer offer
    for t in IDX[BYR][:-1]:
        key = 'simple_hours_{}'.format(t)
        feats = get_feats(data=data, turn=t)
        flag = False
        for k, v in feats.items():
            line, bw = ll_wrapper(v[1], v[0], dim=BYR_NORM_DIMS[t])
            print('{}) {}: {}'.format(t, k, bw[0]))

            if not flag:
                line.columns = pd.MultiIndex.from_product([[k], line.columns])
                d[key] = line
                flag = True
            else:
                for col in ['beta', 'err']:
                    d[key].loc[:, (k, col)] = line[col]

    topickle(d, PLOT_DIR + 'byrhours.pkl')


if __name__ == '__main__':
    main()
