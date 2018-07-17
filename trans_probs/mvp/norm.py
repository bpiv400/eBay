# load packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse
import pickle


def get_resp_offr(turn):
    '''
    Description: Determines the name of the response column given the name of the last observed turn
    for offer models
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = 'b0'
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
    return resp_col


def get_resp_time(turn):
    '''
    Description: Determines the name of the response column given the name of the last observed turn
    for time models
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = 'b0'
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'time_%s' % resp_turn
    return resp_col


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', action='store', type=str)
    # gives the turn immediately previous to the turn this data set
    # is used to predict (ie the last observed turn)
    parser.add_argument('--turn', action='store', type=str)
    # gives the name of the current experiment
    parser.add_argument('--exp', action='store', type=str)

    # parse args
    args = parser.parse_args()
    name = args.name
    turn = args.turn.strip()
    exp_name = args.exp

    # load data frame
    load_loc = 'data/exps/%s/binned/%s_concat_%s.csv' % (exp_name, name, turn)
    df = pd.read_csv(load_loc)

    # find response column names
    resp_offr = get_resp_offr(turn)
    resp_time = get_resp_time(turn)

    # also temporarily extract all reference columns and the response column
    extract_cols = ['ref_rec', 'ref_old', 'ref_resp']
    extract_cols.append(resp_offr)
    extract_cols.append()

    # create a dictionary to store the columns in temporarily
    # and remove each from the data frame
    temp_dict = {}
    for col in extract_cols:
        if col in df.columns:
            temp_dict[col] = df[col].copy()
            df.drop(columns=col, inplace=True)
        else:
            temp_dict[col] = None

    # check for NaN's existing before normalization
    if df.isna().any().any():
        raise ValueError('NaN existed before normalization')

    # for debugging purposes, check which columns have been removed from
    # the data frame
    for col in temp_dict.keys():
        print('Removed: %s' % col)

    if name == 'train' or name == 'toy':
        # calculate mean and standard deviation for each remaining column
        # as series -- index gives name of the column for both
        mean = df.mean()
        std = df.std()
        std_zeros = std[std == 0].index
        df.drop(columns=std_zeros, inplace=True)
        mean.drop(index=std_zeros, inplace=True)
        std.drop(index=std_zeros, inplace=True)
        df = (df - mean)/std

        # compose mean and std into data frame with 2 columns (mean, std)
        # where each index indicates the corresponding column
        if name == 'train':
            # pickle norm df
            norm_df = pd.DataFrame({'mean': mean, 'std': std})
            norm_df.index.name = 'cols'
            norm_df.to_csv('data/exps/%s/%s/norm.csv' % (exp_name, turn))

            # save index series of std zeros
            std_zeros = pd.Series(std_zeros.values)
            if len(std_zeros.index) == 0:
                std_zeros = pd.Series('NO_ZEROS')
            std_zeros.to_csv('data/exps/%s/%s/zeros.csv' % (exp_name, turn))

    else:
        # load norm df from training corpus
        norm_df = pd.read_csv('data/exps/%s/%s/norm.csv' % (exp_name, turn))
        norm_df.set_index('cols', drop=True, inplace=True)

        # load index series giving names of columns with 0 standard deviation
        std_zeros = pd.read_csv('data/exps/%s/%s/zeros.csv' % (exp_name, turn),
                                squeeze=True, index_col=0, header=None)

        # grab mean and standard deviation
        mean = norm_df['mean']
        std = norm_df['std']

        # drop columns that were dropped
        if 'NO_ZEROS' not in std_zeros.values:
            df.drop(columns=std_zeros, inplace=True)

        df = (df - mean)/std

    # check for NaN's created by normalization
    if df.isna().any().any():
        raise ValueError('Normalization introduced NaN somehwere')
    # add all removed columns back into the data frame
    for col_name, col_ser in temp_dict.items():
        if col_ser is not None:
            df[col_name] = col_ser

    # save the current data frame
    df.to_csv('data/exps/%s/normed/%s_concat_%s.csv' %
              (exp_name, name, turn), index_label=False)


if __name__ == '__main__':
    main()
