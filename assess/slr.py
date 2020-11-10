import pandas as pd
from assess.util import merge_dicts, cdf_days, cdf_sale, norm_dist, ll_wrapper, \
    arrival_dist, hist_dist, delay_dist, con_dist, num_threads, num_offers
from agent.util import find_best_run, get_slr_valid, load_valid_data, \
    get_norm_reward, load_values
from utils import topickle, load_data, safe_reindex
from agent.const import COMMON_CONS
from assess.const import DELTA_SLR, NORM1_DIM
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
        print('Turn {0:d} reject rate: {1:.3f} ({2:.3f} exp)'.format(
            turn, (con == 0).mean(), exp.mean()))
        print('Turn {0:d} accept rate: {1:.3f}'.format(
            turn, (con == 1).mean()))
        print('Turn {0:d} concession rate: {1:.3f}'.format(
            turn, ((con > 0) & (con < 1)).mean()))


def get_feats(data=None, values=None, feat=None):
    norm = data[X_OFFER].loc[~data[X_OFFER][AUTO] & ~data[X_OFFER][EXP], NORM]
    idx = norm.xs(2, level=INDEX).index
    norm1 = norm.xs(1, level=INDEX).loc[idx]

    # throw out small opening concessions (helps with bandwidth estimation)
    norm1 = norm1[norm1 > .33]
    idx = norm1.index

    # sale norm and continuation value
    sale_norm, cont_value = get_norm_reward(data=data,
                                            values=(DELTA_SLR * values))

    if feat == 'sale':
        sale_norm = safe_reindex(sale_norm, idx=idx)
        sale_norm[sale_norm.isna()] = 0
        return norm1.values, sale_norm.values
    elif feat == 'reward':
        reward = pd.concat([sale_norm, cont_value]).sort_index().rename('reward')
        reward = safe_reindex(reward, idx=idx)
        return norm1.values, reward.values
    else:
        raise ValueError('Invalid feat name: {}'.format(feat))


def wrapper(data=None, values=None, feat=None, bw=None):
    x, y = get_feats(data=data, values=values, feat=feat)
    if bw is None:
        line, dots, bw = ll_wrapper(y, x,
                                    dim=NORM1_DIM,
                                    discrete=COMMON_CONS[1])
        return line, dots, bw
    else:
        line, dots, _ = ll_wrapper(y, x,
                                   dim=NORM1_DIM,
                                   discrete=COMMON_CONS[1],
                                   bw=bw, ci=False)
        return line, dots


def compare_rewards(data_obs=None, data_rl=None, values=None, feat=None):
    # observed data
    line, dots, bw = wrapper(data=data_obs, values=values, feat=feat)
    line.columns = pd.MultiIndex.from_product([['Data'], line.columns])
    dots.columns = pd.MultiIndex.from_product([['Data'], dots.columns])
    tup = line, dots
    print('{}: {}'.format(feat, bw[0]))

    line, dots = wrapper(data=data_rl, values=values, feat=feat, bw=bw)
    tup[0].loc[:, ('Agent', 'beta')] = line
    tup[1].loc[:, ('Agent', 'beta')] = dots

    return tup


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
    # data
    values = load_values(part=TEST, delta=DELTA_SLR)
    data_obs = get_slr_valid(load_data(part=TEST))
    run_dir = find_best_run(byr=False, delta=DELTA_SLR)
    data_rl = load_valid_data(part=TEST, run_dir=run_dir)

    # descriptives
    d = collect_outputs(data=data_obs, name='Data')
    d_rl = collect_outputs(data=data_rl, name='Agent')
    d = merge_dicts(d, d_rl)

    # reward comparison
    for feat in ['sale', 'reward']:
        key = 'response_{}norm'.format(feat)
        d[key] = compare_rewards(data_obs=data_obs,
                                 data_rl=data_rl,
                                 values=values,
                                 feat=feat)

    # save
    topickle(d, PLOT_DIR + 'slr.pkl')


if __name__ == '__main__':
    main()
