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
        if name == 'Agent':
            con_rate = con.groupby(con).count() / len(con)
            con_rate = con_rate[con_rate > .001]
            print(con_rate)
        else:
            print('Turn {0:d} avg concession: {1:.3f}'.format(
                turn, con[(con > 0) & (con < 1)].mean()))


def get_feats(data=None, values=None):
    norm = data[X_OFFER].loc[~data[X_OFFER][AUTO] & ~data[X_OFFER][EXP], NORM]
    idx = norm.xs(2, level=INDEX).index
    norm1 = norm.xs(1, level=INDEX).loc[idx]
    # throw out small opening concessions (helps with bandwidth estimation)
    norm1 = norm1[norm1 > .33]
    idx = norm1.index
    # sale norm and reward
    sale_norm, cont_value = get_norm_reward(data=data,
                                            values=(DELTA_SLR * values))
    reward = pd.concat([sale_norm, cont_value]).sort_index().rename('reward')
    reward = safe_reindex(reward, idx=idx)
    sale_norm = safe_reindex(sale_norm, idx=idx)
    sale_norm[sale_norm.isna()] = 0
    return norm1.values, sale_norm.values, reward.values


def collect_outputs(data=None, values=None, name=None, bw=None):
    d = dict()

    if bw is None:
        bw = dict()

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

    x, y_sale, y_reward = get_feats(data=data, values=values)
    for feat in ['sale', 'reward']:
        key = 'response_{}norm'.format(feat)
        y = locals()['y_{}'.format(feat)]

        if name == 'Data':
            line, dots, bw[feat] = ll_wrapper(y, x,
                                              dim=NORM1_DIM,
                                              discrete=COMMON_CONS[1])
            line.columns = pd.MultiIndex.from_product([[name], line.columns])
            dots.columns = pd.MultiIndex.from_product([[name], dots.columns])
            d[key] = line, dots
            print('{}: {}'.format(feat, bw[feat][0]))
        else:
            line, dots, _ = ll_wrapper(y, x,
                                       dim=NORM1_DIM,
                                       discrete=COMMON_CONS[1],
                                       bw=bw[feat], ci=False)
            d[key][0].loc[:, (name, 'beta')] = line
            d[key][1].loc[:, (name, 'beta')] = dots

    # rename series
    for k, v in d.items():
        if type(v) is dict:
            for key, value in v.items():
                d[k][key] = value.rename(name)
        else:
            d[k] = v.rename(name)

    return d


def main():
    values = load_values(part=TEST, delta=DELTA_SLR)

    # observed sellers
    d, bw = collect_outputs(data=get_slr_valid(load_data(part=TEST)),
                            values=values,
                            name='Data')

    # rl seller
    run_dir = find_best_run(byr=False, delta=DELTA_SLR)
    d_rl, _ = collect_outputs(data=load_valid_data(part=TEST, run_dir=run_dir),
                              values=values,
                              name='Agent')

    # concatenate DataFrames
    d = merge_dicts(d, d_rl)

    # save
    topickle(d, PLOT_DIR + 'slr.pkl')


if __name__ == '__main__':
    main()
