from assess.util import merge_dicts, cdf_days, cdf_sale, msg_dist, \
    arrival_dist, hist_dist, delay_dist, con_dist, num_threads, \
    num_offers, interarrival_dist, norm_dist, get_lstgs
from utils import topickle, load_data
from constants import PLOT_DIR
from featnames import X_THREAD, X_OFFER, TEST, SIM


def collect_outputs(data=None, name=None):
    d = dict()

    # sale outcomes
    d['cdf_days'] = cdf_days(data)
    d['cdf_salenorm'], d['cdf_saleprice'] = cdf_sale(data)

    # offer distributions
    d['pdf_arrival'] = arrival_dist(data[X_THREAD])
    d['pdf_interarrival'] = interarrival_dist(data[X_THREAD])
    d['cdf_hist'] = hist_dist(data[X_THREAD])
    d['cdf_delay'] = delay_dist(data[X_OFFER])
    d['cdf_con'] = con_dist(data[X_OFFER])
    d['cdf_norm'] = norm_dist(data[X_OFFER])

    d['bar_msg'] = msg_dist(data[X_OFFER])
    d['bar_threads'] = num_threads(data)
    d['bar_offers'] = num_offers(data[X_OFFER])

    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def main():
    lstgs, suffix = get_lstgs()

    # data
    data_obs = load_data(part=TEST, lstgs=lstgs)
    data_sim = load_data(part=TEST, sim=True, lstgs=lstgs)

    # observed
    d = collect_outputs(data=data_obs, name='Data')

    # simulations
    d_sim = collect_outputs(data=data_sim, name='Simulations')

    # concatenate DataFrames
    d = merge_dicts(d, d_sim)

    # save
    topickle(d, PLOT_DIR + '{}{}.pkl'.format(SIM, suffix))


if __name__ == '__main__':
    main()
