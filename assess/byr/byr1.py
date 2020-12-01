import numpy as np
import pandas as pd
from agent.util import get_norm_reward
from assess.util import ll_wrapper
from utils import load_data, topickle
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM
from constants import PLOT_DIR, INTERVAL_ARRIVAL, INTERVAL_CT_ARRIVAL
from featnames import X_OFFER, X_THREAD, BYR_HIST, CON, INDEX, THREAD, \
    TEST, LOOKUP, CLOCK, START_TIME


def time_to_sale_plot(data=None, values=None):
    is_sale = data[X_OFFER][CON] == 1
    sale_time = data[CLOCK][is_sale].droplevel([THREAD, INDEX])
    interval = (sale_time - data[LOOKUP][START_TIME]) // INTERVAL_ARRIVAL
    interval[interval.isna()] = INTERVAL_CT_ARRIVAL  # no sale
    interval = interval.astype('int64')
    norm_reward = pd.concat(get_norm_reward(data=data, values=values))
    s = pd.Series()
    for i in range(INTERVAL_CT_ARRIVAL + 1):
        s.loc[i] = norm_reward[interval >= i].mean()
    s.index /= INTERVAL_CT_ARRIVAL
    return s


def main():
    d = dict()

    # distributions of byr_hist for those who make 50% concessions
    data = load_data(part=TEST)

    con1 = data[X_OFFER].xs(1, level=INDEX)[CON]
    hist1 = data[X_THREAD][BYR_HIST]
    assert np.all(con1.index == hist1.index)
    y, x = hist1.values, con1.values

    mask = x > .33
    d['response_hist'], bw = ll_wrapper(y[mask], x[mask],
                                        dim=NORM1_DIM,
                                        discrete=COMMON_CONS[1])
    print('hist: {}'.format(bw[0]))

    # duration into listing
    # d['simple_timevals'] = time_to_sale_plot(data=data, values=vals)

    # save
    topickle(d, PLOT_DIR + 'byr1.pkl')


if __name__ == '__main__':
    main()
