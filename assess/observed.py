from assess.util import load_data, get_action_dist, get_lookup, \
    arrival_dist, hist_dist, delay_dist, con_dist
from utils import topickle
from constants import PLOT_DIR, TEST
from featnames import OBS, ARRIVAL, BYR_HIST, DELAY, CON


def construct_d(lookup=None):
    # observed sellers
    data = load_data(part=TEST, lstgs=lookup.index, obs=True)
    threads, offers = [data[k] for k in ['threads', 'offers']]

    d = dict()

    # offer distributions
    d['pdf_{}'.format(ARRIVAL)] = arrival_dist(threads)
    d['cdf_{}'.format(BYR_HIST)] = hist_dist(threads)
    d['cdf_{}'.format(DELAY)] = delay_dist(offers)
    d['cdf_{}'.format(CON)] = con_dist(offers)

    # action probabilities
    d['action'] = get_action_dist(offers_dim=offers, offers_action=offers)

    return d


def main():
    lookup, filename = get_lookup(prefix=OBS)

    # dictionary of inputs for plots
    d = construct_d(lookup)

    # save
    topickle(d, PLOT_DIR + '{}.pkl'.format(filename))


if __name__ == '__main__':
    main()
