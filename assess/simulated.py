from assess.util import load_data, get_action_dist, merge_dicts, count_dist, \
    cdf_days, cdf_sale, get_lstgs, arrival_dist, hist_dist, delay_dist, con_dist
from utils import topickle
from constants import PLOT_DIR, TEST
from featnames import START_PRICE, OBS, SIM, ARRIVAL, BYR_HIST, DELAY, CON


def collect_outputs(data=None, lookup=None, name=None):
    threads, offers, clock = [data[k] for k in ['threads', 'offers', 'clock']]

    d = dict()
    d['cdf_norm'], d['cdf_price'] = cdf_sale(
        offers=offers, start_price=lookup[START_PRICE])

    d['cdf_days'] = cdf_days(offers=offers, clock=clock, lookup=lookup)

    # offer distributions
    d['pdf_{}'.format(ARRIVAL)] = arrival_dist(threads)
    d['cdf_{}'.format(BYR_HIST)] = hist_dist(threads)
    d['cdf_{}'.format(DELAY)] = delay_dist(offers)
    d['cdf_{}'.format(CON)] = con_dist(offers)

    for k in ['threads', 'offers']:
        d['num_{}'.format(k)] = count_dist(data[k], level=k)

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

    # simulated seller
    data[SIM] = load_data(part=TEST, lstgs=lookup.index, sim=True)
    d_sim = collect_outputs(data=data[SIM], lookup=lookup, name='Simulated seller')

    # concatenate DataFrames
    d = merge_dicts(d, d_sim)

    # action probabilities
    d['action'] = get_action_dist(offers_dim=data[OBS]['offers'],
                                  offers_action=data[SIM]['offers'])

    return d


def main():
    lookup, filename = get_lstgs(prefix=SIM)

    # dictionary of inputs for plots
    d = construct_d(lookup)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
