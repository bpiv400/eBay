from analyze.util import merge_dicts, delay_dist, cdf_sale, con_dist, \
    num_offers, thread_number, arrival_cdf, hist_dist, save_dict, rename_series
from agent.util import get_sim_dir, load_valid_data
from featnames import DELAY, X_OFFER, X_THREAD, CON


def collect_outputs(data=None):
    d = dict()

    # offer distributions
    d['cdf_lstgnorm'], d['cdf_lstgprice'] = cdf_sale(data, sales=False)
    d['cdf_arrivaltime'] = arrival_cdf(data[X_THREAD])
    d['cdf_hist'] = hist_dist(data[X_THREAD])
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])
    d['cdf_{}'.format(CON)] = con_dist(data[X_OFFER])
    d['bar_threads'] = thread_number(data[X_THREAD])
    d['bar_offers'] = num_offers(data[X_OFFER])

    return d


def main():
    # observed buyers
    data = load_valid_data(byr=True, minimal=True)
    d = rename_series(collect_outputs(data=data), name='Humans')

    # rl buyer
    sim_dir = get_sim_dir(byr=True, delta=1)
    data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)
    d_rl = rename_series(collect_outputs(data=data_rl), name='$\\lambda = 1$ agent')
    d = merge_dicts(d, d_rl)

    # save
    save_dict(d, 'byr')


if __name__ == '__main__':
    main()
