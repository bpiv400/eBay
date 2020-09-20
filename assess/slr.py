import numpy as np
from assess.util import load_data, get_action_dist, merge_dicts, count_dist, cdf_days, cdf_sale, get_lookup, arrival_dist, \
    hist_dist, delay_dist, con_dist, norm_norm, accept3d
from agent.util import find_best_run, get_valid_slr
from utils import topickle
from constants import PLOT_DIR, TEST
from featnames import SLR, START_PRICE, OBS, RL, ARRIVAL, BYR_HIST, DELAY, CON, \
    ACCEPT, NORM


def collect_outputs(data=None, lookup=None, name=None):
    data, lookup = get_valid_slr(data=data, lookup=lookup)  # restrict to valid listings
    threads, offers, clock = [data[k] for k in ['threads', 'offers', 'clock']]

    d = dict()
    d['cdf_norm'], d['cdf_price'] = cdf_sale(
        offers=offers,
        start_price=lookup[START_PRICE]
    )

    d['cdf_days'] = cdf_days(offers=offers, clock=clock, lookup=lookup)

    # offer distributions
    d['pdf_{}'.format(ARRIVAL)] = arrival_dist(threads)
    d['cdf_{}'.format(BYR_HIST)] = hist_dist(threads)
    d['cdf_{}'.format(DELAY)] = delay_dist(offers)
    d['cdf_{}'.format(CON)] = con_dist(offers)

    # norm-norm plot
    d['norm-norm'] = norm_norm(offers[NORM])

    # thread and offer counts
    for k in ['threads', 'offers']:
        d['num_{}'.format(k)] = count_dist(data[k], level=k)

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def construct_d(lookup=None):
    data = dict()

    # observed sellers
    data[OBS] = load_data(part=TEST, lstgs=lookup.index, obs=True)
    d = collect_outputs(data=data[OBS], lookup=lookup, name='Observed sellers')

    # RL seller
    data[RL] = load_data(part=TEST, lstgs=lookup.index, run_dir=find_best_run())
    d_rl = collect_outputs(data=data[RL], lookup=lookup, name='RL seller')

    # concatenate DataFrames
    d = merge_dicts(d, d_rl)

    # accept probabilities
    d[ACCEPT], other = dict(), np.log10(lookup[START_PRICE])
    for k, v in data.items():
        d[ACCEPT][k] = accept3d(offers=v['offers'], other=other)

    # action probabilities
    # d['action'] = get_action_dist(offers_dim=data[OBS]['offers'],
    #                               offers_action=data[RL]['offers'],
    #                               byr=False)

    return d


def main():
    lookup, filename = get_lookup(prefix=SLR)

    # dictionary of inputs for plots
    d = construct_d(lookup)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
