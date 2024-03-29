import argparse
from analyze.util import merge_dicts, cdf_days, cdf_sale, msg_dist, \
    arrival_dist, hist_dist, delay_dist, con_dist, num_threads, \
    num_offers, interarrival_dist, norm_dist, save_dict, rename_series
from utils import load_data, load_file
from analyze.const import COLLECTIBLES
from featnames import X_THREAD, X_OFFER, TEST, SIM, LOOKUP, STORE, START_PRICE, META


def collect_outputs(data=None):
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

    return d


def get_lstgs():
    # subset from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str)
    subset = parser.parse_args().subset

    if subset is None:
        return None, ''

    # restrict listings
    feats = load_file(TEST, LOOKUP)
    if subset == 'store':
        lstgs = feats[feats[STORE]]
    elif subset == 'no_store':
        lstgs = feats[~feats[STORE]]
    elif subset == 'price_low':
        lstgs = feats[feats[START_PRICE] < 99]
    elif subset == 'price_high':
        lstgs = feats[feats[START_PRICE] >= 99]
    elif subset == 'collectibles':
        lstgs = feats[feats[META].apply(lambda x: x in COLLECTIBLES)]
    elif subset == 'other':
        lstgs = feats[feats[META].apply(lambda x: x not in COLLECTIBLES)]
    else:
        raise NotImplementedError('Unrecognized subset: {}'.format(subset))
    print('{}: {} listings'.format(subset, len(lstgs)))

    return lstgs.index, '_{}'.format(subset)


def main():
    lstgs, suffix = get_lstgs()

    # data
    data_obs = load_data(lstgs=lstgs, clock=True)
    data_sim = load_data(sim=True, lstgs=lstgs, clock=True)

    # observed
    d = rename_series(collect_outputs(data=data_obs), name='Data')

    # simulations
    d_sim = rename_series(collect_outputs(data=data_sim), name='Simulations')

    # concatenate DataFrames
    d = merge_dicts(d, d_sim)

    # save
    save_dict(d, SIM + suffix)


if __name__ == '__main__':
    main()
