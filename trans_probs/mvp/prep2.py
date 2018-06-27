import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse

# to be used with digitize(right = True)
# remove all vals from the associated array that are
# greater than high or less than low (exclusive both)
# before using
# currently doesn't produce a symmetric window around low and high points


def bins_from_midpoints(low, high, step):
    midpoints = np.arange(low, high + step, step)
    odd_bin_cents = midpoints[::2]
    ev_bin_cents = midpoints[1::2]
    if len(midpoints) % 2 == 0:
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        odd_bin_cents = odd_bin_cents[1:]
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high
        bins[:bin_count:2] = low_set
        bins[1:bin_count - 1:2] = high_set
    else:
        odd_bin_cents = odd_bin_cents[:len(odd_bin_cents) - 1]
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        ev_high = ev_bin_cents[len(ev_bin_cents) - 1]
        last_bin = (high + ev_high) / 2
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        odd_bin_cents = odd_bin_cents[1:]
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        high_set = np.append(high_set, last_bin)
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high
        bins[:bin_count - 1:2] = low_set
        bins[1:bin_count:2] = high_set
    return bins, midpoints


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--dir', action='store', type=str)
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--turn', action='store', type=str)
    args = parser.parse_args()
    print(args)
    filename = args.name
    subdir = args.dir
    low = args.low
    high = args.high
    step = args.step
    turn = args.turn.strip()
    if len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)

    # load data slice
    read_loc = 'data/' + subdir + '/' + 'turns/' + turn + '/' + filename
    df = pd.read_csv(read_loc)

    # dropping columns that are not useful for prediction
    df.drop(columns=['anon_item_id', 'anon_thread_id', 'anon_byr_id',
                     'anon_slr_id', 'auct_start_dt',
                     'auct_end_dt', 'item_price', 'bo_ck_yn'], inplace=True)

    # for the timebeing, leave frac_remain and the times observed thusfar
    # as features
    date_list = []
    for i in range(turn_num + 1):

        date_list.append('remain_' + 'b' + str(i))
        date_list.append('passed_' + 'b' + str(i))
        date_list.append('frac_remain_' + 'b' + str(i))
        date_list.append('frac_passed_' + 'b' + str(i))
        date_list.append('date_b' + str(i))
        date_list.append('date_s' + str(i))

        if i < turn_num and turn_type == 'b':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))
        elif turn_type == 's':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))

        if i > 0:
            date_list.append('time_' + 'b' + str(i))
        date_list.append('time_s' + str(i))

        if i == turn_num and turn_type == 'b':
            if ('time_s' + str(i)) not in date_list:
                date_list.append('time_s' + str(i))
        elif i == turn_num and turn_type == 's':
            date_list.append('time_b' + str(i+1))
            date_list.append('date_b' + str(i+1))
    # dropping all unused date and time features
    df.drop(columns=date_list, inplace=True)

    # fixing seller and buyer history
    # assumed that NAN slr and byr histories indicate having participated in
    # 0 best offer threads previously
    # deduced from the fact that no sellers / buyers have 0 as their history
    no_hist_slr = df[np.isnan(df['slr_hist'])].index
    df.loc[no_hist_slr, 'slr_hist'] = 0

    no_hist_byr = df[np.isnan(df['byr_hist'])].index
    df.loc[no_hist_byr, 'byr_hist'] = 0

    del no_hist_byr
    del no_hist_slr

    # setting percent feedback for 'green' sellers to median feedback
    # score
    never_sold = df[np.isnan(df['fdbk_pstv_src'].values)].index
    scores = df['fdbk_pstv_src'].values
    scores = scores[~np.isnan(scores)]
    med_score = np.median(scores)
    df.loc[never_sold, 'fdbk_pstv_src'] = med_score
    del scores
    del med_score
    del never_sold

    # and setting number of previous feedbacks received to 0
    never_sold = df[np.isnan(df['fdbk_score_src'].values)].index
    df.loc[never_sold, 'fdbk_score_src'] = 0
    del never_sold

    # setting initial feedback for green sellers to median
    # feedback score
    never_sold = df[np.isnan(df['fdbk_pstv_start'].values)].index
    scores = df['fdbk_pstv_start'].values
    scores = scores[~np.isnan(scores)]
    med_score = np.median(scores)
    df.loc[never_sold, 'fdbk_pstv_start'] = med_score
    del scores
    del med_score
    del never_sold

    # and setting number of previous feedbacks received to 0
    never_sold = df[np.isnan(df['fdbk_score_start'].values)].index
    df.loc[never_sold, 'fdbk_score_start'] = 0
    del never_sold

    # setting nans for re-listing indicator to 0 because there are
    # no zeros in the indicator column, implying that nans indicate 0
    not_listed = df[np.isnan(df['lstg_gen_type_id'].values)].index
    df.loc[not_listed, 'lstg_gen_type_id'] = 0
    del not_listed

    # setting nans for mssg to 0 arbitrarily since only ~.001% of offers
    # have 0 for message
    no_msg = df[np.isnan(df['any_mssg'].values)].index
    df.loc[no_msg, 'any_mssg'] = 0
    del no_msg

    # dropping columns that have missing values for the timebeing
    # INCLUDING DROPPING decline, accept prices since it feels
    # epistemologically disingenous to use them
    df.drop(columns=['count2', 'count3', 'count4', 'ship_time_fastest', 'ship_time_slowest',
                     'ref_price2', 'ref_price3', 'ref_price4', 'decline_price', 'accept_price'], inplace=True)

    # dropping all threads that do not have ref_price1
    df.drop(df[np.isnan(df['ref_price1'].values)].index, inplace=True)

    # making thread id the index
    df.set_index('unique_thread_id', inplace=True)
    # drop rows with offers below low threshold or start price above high threshhold

    save_loc = 'data/' + 'exps' + \
        '/' + exp_name + '/' turn + '/' + filename
    df.to_csv(save_loc, index_label=False)


if __name__ == '__main__':
    main()
