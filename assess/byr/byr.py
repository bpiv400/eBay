from assess.util import merge_dicts, delay_dist, cdf_sale, get_lstgs
from agent.util import find_best_run, load_valid_data, only_byr_agent
from utils import topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR
from featnames import DELAY, X_OFFER, TEST, BYR


def collect_outputs(data=None, name=None):
    d = dict()

    # offer distributions
    d['cdf_lstgnorm'], d['cdf_lstgprice'] = cdf_sale(data, sales=False)
    d['cdf_{}'.format(DELAY)] = delay_dist(data[X_OFFER])

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

    # observed buyers
    data = only_byr_agent(load_valid_data(part=TEST, byr=True, lstgs=lstgs))
    d = collect_outputs(data=data, name='Data')

    # rl buyer
    for delta in DELTA_BYR:
        run_dir = find_best_run(byr=True, delta=delta)
        data_rl = load_valid_data(
            part=TEST, run_dir=run_dir, byr=True, lstgs=lstgs)
        data_rl = only_byr_agent(data_rl)
        d_rl = collect_outputs(
            data=data_rl, name='$\\delta = {}$'.format(delta))
        d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + '{}{}.pkl'.format(BYR, suffix))


if __name__ == '__main__':
    main()
