import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import kreg2, ll_wrapper
from utils import topickle, safe_reindex
from assess.const import NORM2_BIN_MESH, POINTS
from constants import PLOT_DIR
from featnames import X_OFFER, CON, NORM, TEST, INDEX, LOOKUP, START_PRICE

DIM = {2: np.linspace(.65, .95, POINTS)}


def get_feats(data=None, turn=None):
    con = data[X_OFFER][CON].xs(turn + 1, level=INDEX)
    x1 = 1 - data[X_OFFER][NORM].xs(turn, level=INDEX).loc[con.index].values
    x2 = np.log10(safe_reindex(data[LOOKUP][START_PRICE], idx=con.index).values)
    y = con.values
    return x1, x2, y


def main():
    d = dict()

    data_obs = only_byr_agent(load_valid_data(part=TEST, byr=True))
    data_rl = only_byr_agent(load_valid_data(part=TEST, run_dir=get_run_dir()))

    for turn in [2]:
        x1_obs, x2_obs, y_obs = get_feats(data=data_obs, turn=turn)
        x1_rl, x2_rl, y_rl = get_feats(data=data_rl, turn=turn)

        # walk rate
        key = 'offer{}walk'.format(turn)
        line, dots, bw = ll_wrapper(y=(y_obs == 0), x=x1_obs,
                                    discrete=[1.], dim=DIM[turn])
        print('{}: {}'.format(key, bw[0]))
        d['response_{}'.format(key)] = line, dots

        # accept rate
        key = 'offer{}acc'.format(turn)
        line, dots, bw = ll_wrapper(y=(y_obs == 1), x=x1_obs,
                                    discrete=[1.], dim=DIM[turn])
        print('{}: {}'.format(key, bw[0]))
        for obj in [line, dots]:
            obj.columns = pd.MultiIndex.from_product([['Humans'], obj.columns])
        d['response_{}'.format(key)] = line, dots

        line, dots, _ = ll_wrapper(y=(y_rl == 1), x=x1_rl,
                                   discrete=[0.], dim=DIM[turn], bw=bw, ci=False)
        d['response_{}'.format(key)][0].loc[:, ('Agent', 'beta')] = line
        d['response_{}'.format(key)][1].loc[:, ('Agent', 'beta')] = dots

        # k = 'offer{}binwalk'.format(turn)
        # d['contour_{}_data'.format(k)], bw = \
        #     kreg2(y=(y_obs == 0), x1=x1_obs, x2=x2_obs, mesh=NORM2_BIN_MESH)
        # print('{}: {}'.format(k, bw))
        #
        # k = 'offer{}binacc'.format(turn)
        # d['contour_{}_data'.format(k)], bw = \
        #     kreg2(y=(y_obs == 1), x1=x1_obs, x2=x2_obs, mesh=NORM2_BIN_MESH)
        # print('{}: {}'.format(k, bw))
        #
        # d['contour_{}_agent'.format(k)], _ = \
        #     kreg2(y=(y_rl == 1), x1=x1_rl, x2=x2_rl,
        #           mesh=NORM2_BIN_MESH, bw=bw)

    topickle(d, PLOT_DIR + 'byrresponse.pkl')


if __name__ == '__main__':
    main()
