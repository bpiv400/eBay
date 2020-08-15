import pandas as pd
from assess.util import load_data, get_valid_slr, get_action_dist, find_best_run, \
    fill_nans, count_dist, cdf_months, cdf_sale, get_lookup
from utils import topickle
from constants import PLOT_DIR, TEST, AGENT_DIR
from featnames import SLR, AUTO, START_PRICE, OBS, RL


def collect_outputs(data=None, lookup=None, name=None):
    valid = get_valid_slr(data['offers'][AUTO])

    d = dict()
    d['cdf_norm'], d['cdf_price'] = cdf_sale(
        offers=data['offers'],
        start_price=lookup[START_PRICE]
    )

    d['cdf_norm_eligible'], d['cdf_price_eligible'] = cdf_sale(
        offers=data['offers'],
        start_price=lookup[START_PRICE],
        valid=valid
    )

    d['cdf_months'] = cdf_months(
        offers=data['offers'],
        clock=data['clock'],
        lookup=lookup,
        valid=valid
    )

    for k in ['threads', 'offers']:
        d['num_{}'.format(k)] = count_dist(data[k], level=k, valid=valid)

    for k, v in d.items():
        d[k] = v.rename(name)

    return d


def construct_d(lookup=None):
    data = dict()

    # observed sellers
    data[OBS] = load_data(part=TEST, lstgs=lookup.index, obs=True)
    d = collect_outputs(data=data[OBS], lookup=lookup, name='Observed sellers')

    # RL seller
    # run_dir = find_best_run()
    run_dir = AGENT_DIR + '{}/run_entropy_0.1_dollar/'.format(SLR)
    data[RL] = load_data(part=TEST, lstgs=lookup.index, run_dir=run_dir)
    d_rl = collect_outputs(data=data[RL], lookup=lookup, name='RL seller')

    # concatenate DataFrames
    for k, v in d.items():
        d[k] = pd.concat([v, d_rl[k]], axis=1, sort=True)
        if d[k].isna().sum().sum() > 0:
            d[k] = fill_nans(d[k])

    # action probabilities
    d['action'] = get_action_dist(data=data, dim_key=OBS)

    return d


def main():
    lookup, filename = get_lookup(prefix=SLR)

    # dictionary of inputs for plots
    d = construct_d(lookup)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
