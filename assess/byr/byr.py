from assess.util import merge_dicts, delay_dist, cdf_sale, \
    con_dist, num_offers
from agent.util import get_run_dir, load_valid_data, only_byr_agent
from utils import topickle
from constants import PLOT_DIR
from featnames import DELAY, X_OFFER, TEST, CON, LOOKUP


def collect_outputs(data=None, name=None):
    d = dict()

    # offer distributions
    d['cdf_lstgnorm'], d['cdf_lstgprice'] = cdf_sale(data, sales=False)
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])
    d['cdf_{}'.format(CON)] = con_dist(data[X_OFFER])
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
    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    d = collect_outputs(data=data, name='Humans')

    # rl buyer
    run_dir = get_run_dir()
    data_rl = load_valid_data(part=TEST,
                              run_dir=run_dir,
                              lstgs=data[LOOKUP].index)
    data_rl = only_byr_agent(data_rl)
    d_rl = collect_outputs(data=data_rl, name='Agent')
    d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'byr.pkl')


if __name__ == '__main__':
    main()
