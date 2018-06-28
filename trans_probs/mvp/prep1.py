import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse


def main():
    '''
    Description: imputes variables for nan values when reasonable to do so,
    removes all other nan columns, deletes all columns considered epistomological 
    cheating (from the point of view of the buyer) for the time being, 
    deletes all columns offers that do not have a reference price, 
    deletes all date features except for observed time_ji features
    and frac_remain_ji for observed turns

    Input: See parameters from argparse
    Output: data chunks prepped for final concatenation, binning, then training
    '''
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    # subdirectory name, corresponding to the type of file being pre-processed
    # (toy, train, test, etc.)
    parser.add_argument('--dir', action='store', type=str)
    # name of the file we're pre-processing, should be
    # 'subdir-n.csv'
    parser.add_argument('--name', action='store', type=str)
    # turn name of the last offer made before the prediction variable
    # for this data set
    parser.add_argument('--turn', action='store', type=str)
    # name of the experiment
    parser.add_argument('--exp', action='store', type=str)
    # parse arguments
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    turn = args.turn.strip()
    exp_name = args.exp
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

    # creating a list of date features that should be dropped
    # in this case, we're dropping all date features encountered
    # except for observed time_ji and frac_remain_ji features
    date_list = []
    for i in range(turn_num + 1):
        # up to and including the current turn number, drop all
        # buyer date features (except time)
        # and date for buyer and seller turns
        # the date for all buyer and seller feautures is present
        # up to and including the current turn because we include
        # date for the prediction variable in extract_turns
        date_list.append('remain_' + 'b' + str(i))
        date_list.append('passed_' + 'b' + str(i))
        # date_list.append('frac_remain_' + 'b' + str(i))
        date_list.append('frac_passed_' + 'b' + str(i))
        date_list.append('date_b' + str(i))
        # drop all seller dates
        date_list.append('date_s' + str(i))

        if i < turn_num and turn_type == 'b':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            # date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))
        elif turn_type == 's':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            # date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))

        # leaving time features for observed turns
        # if i > 0:
        #     date_list.append('time_' + 'b' + str(i))
        # date_list.append('time_s' + str(i))

        # removing time feature for unobserved seller turn
        if i == turn_num and turn_type == 'b':
            date_list.append('time_s' + str(i))
        # if the prediction variable is a buyer turn,
        # remove the corresonding date features
        elif i == turn_num and turn_type == 's':
            date_list.append('time_b' + str(i+1))
            date_list.append('date_b' + str(i+1))
    # dropping all unused date and time features
    df.drop(columns=date_list, inplace=True)
    # conduct visual inspection of remaining features
    print(df.columns)

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

    # saving cleaned data frame, dropping unique_thread_id
    save_loc = 'data/exps/' + exp_name + \
        '/' + turn + '/' + filename
    df.to_csv(save_loc, index_label=False)


if __name__ == '__main__':
    main()
