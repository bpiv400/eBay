from assess.util import get_action_dist, arrival_dist, hist_dist, delay_dist, con_dist
from utils import topickle, load_data
from constants import PLOT_DIR
from featnames import TEST, X_THREAD, X_OFFER


def main():
    data = load_data(part=TEST)

    d = dict()

    # offer distributions
    d['pdf_arrival'] = arrival_dist(data[X_THREAD])
    d['cdf_hist'] = hist_dist(data[X_THREAD])
    d['cdf_delay'] = delay_dist(data[X_OFFER])
    d['cdf_con'] = con_dist(data[X_OFFER])

    # action probabilities
    d['action'] = get_action_dist(offers_dim=data[X_OFFER],
                                  offers_action=data[X_OFFER])

    # save
    topickle(d, PLOT_DIR + 'obs.pkl')


if __name__ == '__main__':
    main()
