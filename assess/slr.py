from assess.util import merge_dicts, cdf_days, cdf_sale, norm_dist, \
    arrival_dist, hist_dist, delay_dist, con_dist, num_threads, num_offers
from agent.util import find_best_run, get_slr_valid, load_valid_data
from utils import topickle, load_data
from agent.const import DELTA_CHOICES
from constants import PLOT_DIR
from featnames import ARRIVAL, DELAY, CON, X_OFFER, X_THREAD, TEST, AUTO, INDEX, \
    EXP, NORM


def print_summary(data=None, name=None):
    print(name)
    for turn in [2, 4, 6]:
        # find valid indices
        is_turn = data[X_OFFER].index.get_level_values(INDEX) == turn
        idx = data[X_OFFER][~data[X_OFFER][AUTO] & is_turn].index
        con = data[X_OFFER].loc[idx, CON]
        exp = data[X_OFFER].loc[idx, EXP]
        con.loc[exp] = -.1
        print('Turn {0:d} reject rate: {1:.4f} ({2:.4f} exp)'.format(
            turn, (con == 0).mean(), exp.mean()))
        print('Turn {0:d} accept rate: {1:.4f}'.format(
            turn, (con == 1).mean()))
        print('Turn {0:d} concession rate: {1:.4f}'.format(
            turn, ((con > 0) & (con < 1)).mean()))


def collect_outputs(data=None, name=None):
    d = dict()

    # print summary
    print_summary(data=data, name=name)

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

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def main():
    # observed
    data_obs = get_slr_valid(load_data(part=TEST))
    d = collect_outputs(data=data_obs, name='Data')

    # seller runs
    for delta in DELTA_CHOICES:
        run_dir = find_best_run(byr=False, delta=delta)
        data_rl = load_valid_data(part=TEST, run_dir=run_dir)
        d_rl = collect_outputs(data=data_rl,
                               name='Agent: $\\delta = {}$'.format(delta))
        d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'slr.pkl')


if __name__ == '__main__':
    main()
