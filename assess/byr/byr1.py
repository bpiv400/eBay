import numpy as np
from agent.util import only_byr_agent, load_valid_data
from assess.util import ll_wrapper
from utils import topickle
from assess.const import NORM1_DIM_LONG
from constants import PLOT_DIR
from featnames import X_OFFER, X_THREAD, BYR_HIST, CON, INDEX, TEST


def main():
    d = dict()

    # distributions of byr_hist for those who make 50% concessions
    data = only_byr_agent(load_valid_data(part=TEST, byr=True))

    con = data[X_OFFER].xs(1, level=INDEX)[CON]
    hist = data[X_THREAD][BYR_HIST]
    assert np.all(con.index == hist.index)
    y, x = hist.values, con.values

    mask = x > .3
    line, dots, bw = ll_wrapper(y[mask], x[mask],
                                dim=NORM1_DIM_LONG,
                                discrete=[.5, 1])
    d['response_hist'] = line, dots
    print('hist: {}'.format(bw[0]))

    # save
    topickle(d, PLOT_DIR + 'byr1.pkl')


if __name__ == '__main__':
    main()
