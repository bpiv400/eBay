from assess.util import merge_dicts, cdf_days, cdf_sale, norm_dist, \
    arrival_dist, hist_dist, delay_dist, con_dist, num_threads, \
    num_offers, get_lstgs
from agent.util import get_run_dir, load_valid_data
from utils import topickle
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import PLOT_DIR, IDX
from featnames import ARRIVAL, DELAY, CON, X_OFFER, X_THREAD, TEST, AUTO, \
    INDEX, NORM, SLR, REJECT


def reject_rate(offers=None):
    offers = offers.loc[~offers[AUTO] & offers.index.isin(IDX[SLR], level=INDEX)]
    return (offers[CON] == 0).groupby(INDEX).mean()


def collect_outputs(data=None, name=None):
    d = dict()

    # sales
    d['cdf_days'] = cdf_days(data)
    d['cdf_lstgnorm'], d['cdf_lstgprice'] = cdf_sale(data, sales=False)

    # offer distributions
    d['pdf_{}'.format(ARRIVAL)] = arrival_dist(data[X_THREAD])
    d['cdf_{}'.format('hist')] = hist_dist(data[X_THREAD])
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])
    d['cdf_{}'.format(CON)] = con_dist(data[X_OFFER])
    d['cdf_{}'.format(NORM)] = norm_dist(data[X_OFFER])

    # thread and offer counts
    d['bar_threads'] = num_threads(data)
    d['bar_offers'] = num_offers(data[X_OFFER])

    # turn-specific active rejection rates
    d['bar_{}'.format(REJECT)] = reject_rate(data[X_OFFER])

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def main():
    lstgs, suffix = get_lstgs()

    # observed
    data_obs = load_valid_data(part=TEST, byr=False, lstgs=lstgs)
    d = collect_outputs(data=data_obs, name='Humans')

    # seller runs
    for delta in DELTA_SLR:
        run_dir = get_run_dir(byr=False, delta=delta)
        data_rl = load_valid_data(part=TEST, run_dir=run_dir, lstgs=lstgs)
        d_rl = collect_outputs(data=data_rl, name=SLR_NAMES[delta])
        d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'slr_{}.pkl'.format(suffix))


if __name__ == '__main__':
    main()