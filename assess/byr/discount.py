import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import continuous_cdf, ll_wrapper
from utils import topickle, safe_reindex
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, NORM, TEST, LOOKUP, START_PRICE


def cdf_wrapper(s):
    return pd.concat([continuous_cdf(s[c]).rename('Turn {}'.format(c))
                      for c in s.columns], axis=1).ffill()


def get_feats(data=None):
    norm = data[X_OFFER][NORM].unstack()
    norm.loc[norm[2].isna(), 2] = 0.
    discount = norm[[2, 4, 6]]
    discount.loc[discount[4].isna(), 4] = discount.loc[discount[4].isna(), 2]
    discount.loc[discount[6].isna(), 6] = discount.loc[discount[6].isna(), 4]
    start_price = safe_reindex(data[LOOKUP][START_PRICE], idx=discount.index)
    dollars = pd.concat([(discount[c] * start_price).rename(c)
                         for c in discount.columns], axis=1)
    return discount, dollars, start_price


def main():
    d = dict()

    # humans
    data_obs = only_byr_agent(load_valid_data(part=TEST, byr=True))
    discount_obs, dollars_obs, start_price_obs = get_feats(data_obs)
    d['cdf_discount_data'] = cdf_wrapper(discount_obs)
    d['cdf_totaldiscount_data'] = cdf_wrapper(dollars_obs)

    # agent
    data_rl = only_byr_agent(load_valid_data(part=TEST, run_dir=get_run_dir()))
    discount_rl, dollars_rl, start_price_rl = get_feats(data_rl)
    d['cdf_discount_agent'] = cdf_wrapper(discount_rl)
    d['cdf_totaldiscount_agent'] = cdf_wrapper(dollars_rl)

    # discount ~ start price
    bw = dict()
    x_obs, x_rl = np.log10(start_price_obs).values, np.log10(start_price_rl).values
    df_obs, df_rl = pd.DataFrame(), pd.DataFrame()
    for t in [2, 4, 6]:
        c = 'Turn {}'.format(t)
        df_obs[c], bw[t] = ll_wrapper(y=discount_obs[t].values, x=x_obs,
                                      dim=LOG10_BIN_DIM, ci=False)
        print('Turn {}: {}'.format(t, bw[t][0]))
        df_rl[c], _ = ll_wrapper(y=discount_rl[t].values, x=x_rl,
                                 dim=LOG10_BIN_DIM, ci=False, bw=bw[t])
    d['simple_discount_data'] = df_obs
    d['simple_discount_agent'] = df_rl

    # bar charts of average discount
    d['bar_discount'] = pd.concat([discount_obs.mean().rename('Humans'),
                                   discount_rl.mean().rename('Agent')], axis=1)
    d['bar_totaldiscount'] = pd.concat([dollars_obs.mean().rename('Humans'),
                                        dollars_rl.mean().rename('Agent')], axis=1)

    topickle(d, PLOT_DIR + 'byrdiscount.pkl')


if __name__ == '__main__':
    main()
