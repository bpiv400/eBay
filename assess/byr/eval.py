import numpy as np
import pandas as pd
from agent.util import load_valid_data, get_run_dir, only_byr_agent, get_sale_norm
from assess.util import get_eval_df, bin_vs_reward, ll_wrapper
from utils import topickle
from assess.const import LOG10_BIN_DIM
from constants import PLOT_DIR
from featnames import TEST, LOOKUP, X_OFFER, START_PRICE, CON, INDEX


def get_discount(data=None):
    sale_norm = get_sale_norm(offers=data[X_OFFER], drop_thread=True)
    discount = (1 - sale_norm).reindex(index=data[LOOKUP].index, fill_value=0)
    discount *= data[LOOKUP][START_PRICE]
    return discount


def get_num_offers(data=None):
    con = data[X_OFFER][CON]
    con = con[con.index.isin([1, 3, 5], level=INDEX)]
    con = con[(0 < con) & (con < 1)]
    num_offers = con.groupby(con.index.names[:-2]).count()
    num_offers = num_offers.reindex(index=data[LOOKUP].index, fill_value=0)
    return num_offers


def bin_vs_offers(data=None, bw=None):
    y = get_num_offers(data).values
    x = np.log10(data[LOOKUP][START_PRICE].values)
    line, bw = ll_wrapper(y=y,
                          x=x,
                          dim=LOG10_BIN_DIM,
                          bw=bw,
                          ci=(bw is None))
    return line, bw


def bin_vs_avg_discount(data=None, bw=None):
    num_offers = get_num_offers(data)
    discount = get_discount(data)
    idx = num_offers[num_offers > 0].index
    y = (discount / num_offers).loc[idx].values
    x = np.log10(data[LOOKUP][START_PRICE].loc[idx].values)
    line, bw = ll_wrapper(y=y,
                          x=x,
                          dim=LOG10_BIN_DIM,
                          bw=bw,
                          ci=(bw is None))
    return line, bw


def compare(data_obs=None, data_rl=None, f=None):
    line, bw = f(data_obs, None)
    line.columns = pd.MultiIndex.from_product([['Humans'], line.columns])
    print('bw: {}'.format(bw[0]))
    line.loc[:, ('Agent', 'beta')], _ = f(data_rl, bw)
    return line


def main():
    d = dict()

    # load data
    data_obs = only_byr_agent(load_valid_data(part=TEST, byr=True))
    data_rl = only_byr_agent(load_valid_data(part=TEST,
                                             run_dir=get_run_dir(),
                                             lstgs=data_obs[LOOKUP].index))

    # discount ~ list price
    d['simple_discountbin'] = compare(
        data_obs=data_obs,
        data_rl=data_rl,
        f=lambda data, bw: bin_vs_reward(data=data, bw=bw, byr=True)
    )

    # num offer ~ list price
    d['simple_offersbin'] = compare(
        data_obs=data_obs,
        data_rl=data_rl,
        f=bin_vs_offers
    )

    # discount per offer ~ list price
    d['simple_avgdiscountbin'] = compare(
        data_obs=data_obs,
        data_rl=data_rl,
        f=bin_vs_avg_discount
    )

    # bar chart of reward
    df = get_eval_df()
    for c in df.columns:
        d['bar_{}'.format(c)] = df[c]

    topickle(d, PLOT_DIR + 'byreval.pkl')


if __name__ == '__main__':
    main()
