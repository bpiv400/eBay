from assess.util import merge_dicts, delay_dist
from agent.util import find_best_run, load_valid_data, get_byr_agent
from utils import topickle, load_data, safe_reindex
from constants import PLOT_DIR
from featnames import DELAY, X_OFFER, CLOCK, TEST


def collect_outputs(data=None, name=None):
    d = dict()

    # offer distributions
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
    # observed buyers
    d = collect_outputs(data=load_data(part=TEST), name='Data')

    # rl buyer
    run_dir = find_best_run(byr=True, delta=.9)
    data = load_valid_data(part=TEST, run_dir=run_dir)
    byr_agent = get_byr_agent(data)
    for k in [X_OFFER, CLOCK]:
        data[k] = safe_reindex(data[k], idx=byr_agent)
    d_rl = collect_outputs(data=data, name='Agent')

    # concatenate DataFrames
    d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'byr.pkl')


if __name__ == '__main__':
    main()
