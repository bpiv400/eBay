import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    # subdirectory corresponding to the type of file we're preprocessing
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
    exp_name = args.exp
    turn = args.turn.strip()
    # verify turn argument is of the correct form
    if len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)

    # load data slice
    read_loc = 'data/' + subdir + '/' + 'turns/' + turn + '/' + filename
    df = pd.read_csv(read_loc, index_col=False)

    # dropping columns that are not useful for prediction
    df.drop(columns=['anon_item_id', 'anon_thread_id', 'anon_byr_id',
                     'anon_slr_id', 'auct_start_dt',
                     'auct_end_dt', 'item_price', 'bo_ck_yn'], inplace=True)

    date_list = []
    # creating a list of date features that should be dropped
    # in this case, we're dropping all date features encountered
    for i in range(turn_num + 1):
        # up to and including the current turn number, drop all
        # buyer date features (except time)
        # and date for buyer and seller turns
        # the date for all buyer and seller feautures is present
        # up to and including the current turn because we include
        # date for the prediction variable in extract_turns
        date_list.append('remain_' + 'b' + str(i))
        date_list.append('passed_' + 'b' + str(i))
        date_list.append('frac_remain_' + 'b' + str(i))
        date_list.append('frac_passed_' + 'b' + str(i))
        date_list.append('date_b' + str(i))
        date_list.append('date_s' + str(i))

        # up to but not including the current turn, if
        # the final turn made is a buyer turn, remove all
        # seller turn date features
        # we exclude the current turn because we have yet
        # to observe seller turn i on buyer turn i,
        # so these features have been excluded from the data
        if i < turn_num and turn_type == 'b':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))
        # if the current turn is a seller turn, remove all
        # date features for sellers up to and including the
        # current turn
        elif turn_type == 's':
            date_list.append('remain_' + 's' + str(i))
            date_list.append('passed_' + 's' + str(i))
            date_list.append('frac_remain_' + 's' + str(i))
            date_list.append('frac_passed_' + 's' + str(i))
        # if we're not currently observing the first turn,
        # remove the time associated with the buyer turn
        if i > 0:
            date_list.append('time_' + 'b' + str(i))
        # remove the time associated with all seller turns
        # up to and including the current turn (since
        # we include time_si even when predicting
        # time si
        date_list.append('time_s' + str(i))

        ############################################################
        # Deprecated --------- doesn't matter in this context where
        # we remove all time variables any
        ##########################################################
        # turn we're currently observing the last turn number
        # and the last turn we've observed is a buyer turn
        #  if i == turn_num and turn_type == 'b':
        #      # then remove the next seller time feature
        #      if ('time_s' + str(i)) not in date_list:
        #          date_list.append('time_s' + str(i))
        ############################################################

        # if we're currently observing the last turn
        # and we're predicting a buyer turn (ie the last observed
        # turn was as seller turn)
        if i == turn_num and turn_type == 's':
            # remove the date and time features for the next buyer turn
            date_list.append('time_b' + str(i+1))
            date_list.append('date_b' + str(i+1))
    # dropping all unused date and time features
    df.drop(columns=date_list, inplace=True)

    # dropping all threads that do not have ref_price1
    df.drop(df[np.isnan(df['ref_price1'].values)].index, inplace=True)

    # remove ship time columns because they trip up the logic later
    df.drop(columns=['ship_time_slowest', 'ship_time_fastest'], inplace=True)

    # drop all features except for offer history and starting price
    remove_cols = []
    for col in df.columns:
        if ('offr' not in col and
                'date' not in col and
                'time' not in col and
                'remain' not in col and
                'passed' not in col):
            remove_cols.append(col)
    remove_cols.remove('start_price_usd')
    df.drop(columns=remove_cols, inplace=True)

    # visual inspection for correctness
    print(df.columns)

    # write resulting csv to file, dropping unique_thread_id
    save_loc = 'data/' + 'exps' + \
        '/' + exp_name + '/' + turn + '/' + filename
    df.to_csv(save_loc, index=False)


if __name__ == '__main__':
    main()
