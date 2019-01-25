"""
converts csvs into toy data format expected by listings
and rlenv
"""

import pickle
import numpy as np
import pandas as pd


def main():
    """
    Simple main method
    """
    direct = './data/toy/'
    f_consts = '%sconsts.csv' % direct
    f_time = '%stime.csv' % direct
    # load files
    consts = pd.read_csv(f_consts)
    time = pd.read_csv(f_time)
    consts.set_index('item', drop=True, inplace=True)
    # randomly sort time df so it's no longer sorted by time, thread
    time = time.sample(frac=1).reset_index(drop=True)
    time.set_index(['item', 'clock'], inplace=True)
    # define columns exposed to simulator
    simtime = ['num_threads',
               'high_offr',
               'low_slr_offr',
               'remaining',
               'fdbk_pstv_src',
               'fdbk_score_src',
               'slr_hist']

    rltime = ['remaining',
              'fdbk_pstv_src',
              'fdbk_score_src',
              'slr_hist']

    simconsts = ['start_price_usd',
                 'wtchr_count',
                 'slr_us',
                 'photo_count',
                 'to_lst_cnt',
                 'bo_lst_cnt',
                 'view_item_count',
                 'used_ind',
                 'accept_price',
                 'decline_price',
                 'slr_hist',
                 'item_count']

    rlconsts = ['start_price_usd',
                'wtchr_count',
                'slr_us',
                'photo_count',
                'to_lst_cnt',
                'bo_lst_cnt',
                'view_item_count',
                'used_ind',
                'accept_price',
                'decline_price',
                'slr_hist',
                'item_count']

    time_dict = {}
    const_dict = {}
    const_dict['consts'] = consts
    const_dict['simfeats'] = simconsts
    const_dict['rlfeats'] = rlconsts
    time_dict['timedf'] = time
    time_dict['rlfeats'] = rltime
    time_dict['simfeats'] = simtime

    # pickle dictionary
    with open('./data/toy/consts_toy-1.pkl', 'wb') as f:
        pickle.dump(const_dict, f)
    with open('./data/toy/time_toy-1.pkl', 'wb') as f:
        pickle.dump(time_dict, f)


if __name__ == '__main__':
    main()
