# load packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse
import pickle


def get_resp_turn(turn):
    '''
    Description: From a string of length two where first
    digit indicates turn type (seller or buyer) and second
    indicates turn number of the last observed turn, determine
    the prediction value
    '''
    turn_type = turn[0]
    turn_num = turn[1]
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    else:
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
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
    filename = args.name + '_concat.csv'
    turn = args.turn.strip()
    exp_name = args.exp

    # load data frame
    df = pd.read_csv('data/exps/%s/binned/%s_concat_%s.csv' %
                     (exp_name, name, turn))

    # find response column name
    resp_turn = get_resp_turn(turn)

    # temporarily extract response turn to normalize the rest of the data
    temp = df[resp_turn]
    df.drop(columns=[resp_turn], inplace=True)

    if name == 'train' or name == 'toy':
        # calculate mean and standard deviation for each remaining column
        # as series -- index gives name of the column for both
        mean = df.mean()
        std = df.std()
        df = (df - mean)/std

        # add response turn column back
        print(df.dtypes)
        print('This is my theory')
        df[resp_turn] = temp
        print('After')
        del temp

        # compose mean and std into data frame with 2 columns (mean, std)
        # where each index indicates the corresponding column

        if name == 'train':
            # pickle norm df
            norm_df = pd.DataFrame({'mean': mean, 'std': std})
            norm_df.index.name = 'cols'
            norm_df.to_csv('data/exps/%s/%s/norm.csv' % (exp_name, turn))
    else:
        # load norm df from training corpus
        norm_df = pd.read_csv('data/exps/%s/%s/norm.csv' % (exp_name, turn))
        norm_df.set_index('cols', drop=True, inplace=True)
        mean = norm_df['mean']
        std = norm_df['std']
        print(norm_df)
        df = (df - mean)/std
        df[resp_turn] = temp
    # save the current data frame
    df.to_csv('data/exps/%s/normed/%s_concat_%s.csv' %
              (exp_name, name, turn), index_label=False)


if __name__ == '__main__':
    main()
