import numpy as np
import pandas as pd
from agent.util import get_byr_agent, load_values, find_best_run, \
    load_valid_data, get_norm_reward
from assess.util import ll_wrapper, continuous_cdf, kdens_wrapper
from utils import load_data, topickle, safe_reindex
from agent.const import AGENT_CONS, DELTA_CHOICES
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

    # values when buyer arrives
    for delta in DELTA_CHOICES:
        vals = load_values(part=TEST, delta=delta)
        run_dir = find_best_run(byr=True, delta=delta)
        if run_dir is None:
            continue
        data = load_valid_data(part=TEST, run_dir=run_dir)
        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)

        # comparing all to valid values
        kwargs = {'All': vals, 'Valid': valid_vals}
        d['pdf_values_{}'.format(delta)] = kdens_wrapper(**kwargs)

        # comparing first-turn offers to walks
        threads = get_byr_agent(data)
        df = safe_reindex(data[X_OFFER], idx=threads)
        rej1 = (df[CON].xs(1, level=INDEX) == 0).droplevel(THREAD)
        other = rej1[~rej1].index
        reject = rej1[rej1].index
        if len(reject) == 0:
            elem = {'Offer': continuous_cdf(valid_vals.loc[other])}
        elif len(other) == 0:
            elem = {'Reject': continuous_cdf(valid_vals.loc[reject])}
        else:
            elem = {'Reject': continuous_cdf(valid_vals.loc[reject]),
                    'Offer': continuous_cdf(valid_vals.loc[other])}
        d['cdf_t1value_{}'.format(delta)] = pd.DataFrame.from_dict(elem)

    # distributions of byr_hist for those who make 50% concessions
    data = load_data(part=TEST)

    con1 = data[X_OFFER].xs(1, level=INDEX)[CON]
    hist1 = data[X_THREAD][BYR_HIST]
    assert np.all(con1.index == hist1.index)
    y, x = hist1.values, con1.values

    mask = x > .33
    d['response_hist'], bw = ll_wrapper(y[mask], x[mask],
                                        dim=NORM1_DIM,
                                        discrete=AGENT_CONS[1])
    print('hist: {}'.format(bw[0]))

    # duration into listing
    # d['simple_timevals'] = time_to_sale_plot(data=data, values=vals)

    # save
    topickle(d, PLOT_DIR + 'byr1.pkl')


if __name__ == '__main__':
    main()
