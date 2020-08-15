import pandas as pd
from assess.util import load_data, get_action_dist, fill_nans, count_dist, \
    cdf_months, cdf_sale, get_lookup
from utils import topickle
from constants import PLOT_DIR, TEST, DISCRIM_MODEL
from featnames import START_PRICE, OBS, SIM


def collect_outputs(data=None, lookup=None, name=None):
    d = dict()
    d['cdf_norm'], d['cdf_price'] = cdf_sale(
        offers=data['offers'],
        start_price=lookup[START_PRICE]
    )

    d['cdf_months'] = cdf_months(
        offers=data['offers'],
        clock=data['clock'],
        lookup=lookup
    )

    for k in ['threads', 'offers']:
        d['num_{}'.format(k)] = count_dist(data[k], level=k)

    for k, v in d.items():
        d[k] = v.rename(name)

    return d


def construct_d(lookup=None):
    data = dict()

    # observed sellers
    data[OBS] = load_data(part=TEST, lstgs=lookup.index, obs=True)
    d = collect_outputs(data=data[OBS], lookup=lookup, name='Observed sellers')

    # simulated seller
    data[SIM] = load_data(part=TEST, lstgs=lookup.index, sim=True)
    d_rl = collect_outputs(data=data[SIM], lookup=lookup, name='Simulated seller')

    # concatenate DataFrames
    for k, v in d.items():
        d[k] = pd.concat([v, d_rl[k]], axis=1, sort=True)
        if d[k].isna().sum().sum() > 0:
            d[k] = fill_nans(d[k])

    # action probabilities
    d['action'] = get_action_dist(data=data, dim_key=OBS)

    return d


def main():
    lookup, filename = get_lookup(prefix=DISCRIM_MODEL)

    # dictionary of inputs for plots
    d = construct_d(lookup)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
