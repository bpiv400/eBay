import numpy as np
from assess.util import get_action_dist, merge_dicts, count_dist, cdf_days, cdf_sale, get_lstgs, arrival_dist, \
    hist_dist, delay_dist, con_dist, norm_norm, accept3d
from agent.util import find_best_run, get_slr_valid
from utils import topickle, load_data
from constants import PLOT_DIR
from featnames import SLR, START_PRICE, OBS, RL, ARRIVAL, BYR_HIST, DELAY, \
    CON, X_OFFER, LOOKUP, X_THREAD, TEST


def collect_outputs(data=None, name=None):
    data = get_slr_valid(data)  # restrict to valid listings

    d = dict()
    d['cdf_norm'], d['cdf_price'] = cdf_sale(
        offers=data[X_OFFER],
        start_price=data[LOOKUP][START_PRICE]
    )

    d['cdf_days'] = cdf_days(data=data)

    # offer distributions
    d['pdf_{}'.format(ARRIVAL)] = arrival_dist(data[X_THREAD])
    d['cdf_{}'.format('hist')] = hist_dist(data[X_THREAD])
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])
    d['cdf_{}'.format(CON)] = con_dist(data[X_OFFER])

    # norm-norm plot
    # d['norm-norm'] = norm_norm(data[X_OFFER])

    # thread and offer counts
    d['bar_threads'] = count_dist(data[X_THREAD])
    d['bar_offers'] = count_dist(data[X_OFFER])

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def construct_d(lstgs=None):
    data = dict()

    # observed sellers
    data[OBS] = load_data(part=TEST, lstgs=lstgs)
    d = collect_outputs(data=data[OBS], name='Observed sellers')

    # RL seller
    run_dir = find_best_run(byr=False, delta=.7)
    data[RL] = load_data(part=TEST, lstgs=lstgs, run_dir=run_dir)
    d_rl = collect_outputs(data=data[RL], name='RL seller')

    # concatenate DataFrames
    d = merge_dicts(d, d_rl)

    # # accept probabilities
    # d[ACCEPT], other = dict(), np.log10(data[OBS][LOOKUP][START_PRICE])
    # for k, v in data.items():
    #     d[ACCEPT][k] = accept3d(offers=v[X_OFFER], other=other)

    # action probabilities
    # d['action'] = get_action_dist(offers_dim=data[OBS][X_OFFER],
    #                               offers_action=data[RL][X_THREAD],
    #                               byr=False)

    return d


def main():
    lstgs, filename = get_lstgs(prefix=SLR)

    # dictionary of inputs for plots
    d = construct_d(lstgs)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
