import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import ll_wrapper
from utils import topickle
from assess.const import NORM2_DIM_LONG
from constants import PLOT_DIR
from featnames import TEST, X_OFFER, CON, INDEX, NORM


def get_feats(data=None):
    con1 = data[X_OFFER][CON].xs(1, level=INDEX)
    con3 = data[X_OFFER][CON].xs(3, level=INDEX)
    idx = con1[con1 == .5].index.intersection(con3.index)
    con3 = con3.loc[idx]
    norm2 = 1 - data[X_OFFER][NORM].xs(2, level=INDEX).loc[con3.index]
    x = norm2.values
    y = {'con': con3.values,
         'acc': (con3 == 1).values,
         'walk': (con3 == 0).values}
    return y, x


def main():
    d = dict()

    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    y, x = get_feats(data)

    for k, v in y.items():
        line, _ = ll_wrapper(v, x, dim=NORM2_DIM_LONG, bw=(.05,))
        line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
        d['simple_norm2{}3'.format(k)] = line

    run_dir = get_run_dir()
    data = only_byr_agent(load_valid_data(part=TEST, run_dir=run_dir))
    y, x = get_feats(data)

    for k, v in y.items():
        print('{}'.format(k))
        key = 'simple_norm2{}3'.format(k)
        d[key], _ = ll_wrapper(v, x, dim=NORM2_DIM_LONG, bw=(.05,), ci=False)

    topickle(d, PLOT_DIR + 'byr3.pkl')


if __name__ == '__main__':
    main()
