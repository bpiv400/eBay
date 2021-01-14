import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import ll_wrapper
from utils import topickle
from agent.const import TURN_COST_CHOICES, DELTA_BYR
from assess.const import NORM2_DIM_LONG
from constants import PLOT_DIR
from featnames import TEST, X_OFFER, CON, INDEX, NORM


def get_feats(data=None):
    con5 = data[X_OFFER][CON].xs(5, level=INDEX)
    norm4 = 1 - data[X_OFFER][NORM].xs(4, level=INDEX).loc[con5.index]
    x = norm4.values
    y = {'con': con5.values,
         'acc': (con5 == 1).values,
         'walk': (con5 == 0).values}
    return y, x


def main():
    d, bw = dict(), dict()

    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    y, x = get_feats(data)

    for k, v in y.items():
        line, dots, bw[k] = ll_wrapper(v, x, dim=NORM2_DIM_LONG, discrete=[1])
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        dots.columns = pd.MultiIndex.from_product([['Humans'], dots.columns])
        d['response_norm2{}3'.format(k)] = line, dots
        print('{}: {}'.format(k, bw[k][0]))

    for turn_cost in TURN_COST_CHOICES:
        col = '${}'.format(turn_cost)
        run_dir = get_run_dir(byr=True,
                              delta=DELTA_BYR[-1],
                              turn_cost=turn_cost)
        data = load_valid_data(part=TEST, run_dir=run_dir, byr=True)
        if data is None:
            continue
        data = only_byr_agent(data)
        y, x = get_feats(data)

        for k, v in y.items():
            print('{}: {}'.format(col, k))
            key = 'response_norm2{}3'.format(k)
            line, dots, _ = ll_wrapper(v, x,
                                       dim=NORM2_DIM_LONG,
                                       discrete=[1],
                                       bw=bw[k],
                                       ci=False)

            d[key][0].loc[:, (col, 'beta')] = line
            d[key][1].loc[:, (col, 'beta')] = dots

    topickle(d, PLOT_DIR + 'byr5.pkl')


if __name__ == '__main__':
    main()