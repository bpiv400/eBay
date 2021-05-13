from assess.util import merge_dicts, delay_dist, cdf_sale, con_dist, \
    num_offers, thread_number, arrival_cdf
from agent.util import get_sim_dir, load_valid_data
from utils import topickle
from constants import PLOT_DIR
from featnames import DELAY, X_OFFER, X_THREAD, CON


def collect_outputs(data=None, name=None):
    d = dict()

    # offer distributions
    d['cdf_lstgnorm'], d['cdf_lstgprice'] = cdf_sale(data, sales=False)
    d['cdf_arrivaltime'] = arrival_cdf(data[X_THREAD])
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])
    d['cdf_{}'.format(CON)] = con_dist(data[X_OFFER])
    d['bar_threads'] = thread_number(data[X_THREAD])
    d['bar_offers'] = num_offers(data[X_OFFER])

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def main():
    # observed buyers
    data = load_valid_data(byr=True, minimal=True)
    d = collect_outputs(data=data, name='Humans')

    # rl buyer
    sim_dir = get_sim_dir(byr=True, delta=1)
    data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)
    d_rl = collect_outputs(data=data_rl, name='Agent')
    d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'byr.pkl')


if __name__ == '__main__':
    main()
