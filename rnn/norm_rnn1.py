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


def remove_dates(df):
    '''
    Description:
    Removes date columns if any are discovered
    '''
    # extract list of columns:
    cols = df.columns
    # generate list of all date column names
    all_dates = ['date_%s' % code for code in all_offr_codes('b3')]
    # iterate over the list
    for datecol in all_dates:
        # remove the column after checking to ensure it exists
        if datecol in cols:
            df.drop(columns=datecol, inplace=True)
    return df


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


def all_offr_codes(offr_code):
    '''
    Description: Generates a list of offer codes for all offers
    preceeding the offer given by 'offr_code', including that
    offer itself
    Input: String denoting last offer code to be generated
    Returns: list of strings
    '''
    out = []
    # check correct format of offr_code
    if len(offr_code) != 2:
        raise ValueError('offr code should have length 2')
    #  extract turn type and num
    turn_num = int(offr_code[1])
    turn_type = offr_code[0]
    # iterate to (inclusive) turn_num
    for i in range(turn_num + 1):
        # add all buyer turns up to and including the current turn
        # to the list
        out.append('b%d' % i)
        # do not add the seller turn for the last round if
        # the last turn is a buyer turn
        if i < turn_num or turn_type == 's':
            out.append('s%d' % i)
    return out


def get_inc_full_cols(df):
    '''
    Description: Generates a tuple of numpy arrays where the first entry consists
    of all columns in df that contain NA values, and the second consists of
    all columns without NA values
    '''
    # generate boolean series where each index is named after a column
    has_na = df.isna().any()
    # grab columms containing na's
    inc_cols = has_na[has_na].index.values
    # grab cols not containing na's
    full_cols = has_na[~has_na].index.values
    return inc_cols, full_cols


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', action='store', type=str)
    # gives the name of the current experiment
    parser.add_argument('--exp', action='store', type=str)

    # parse args
    args = parser.parse_args()
    name = args.name
    exp_name = args.exp

    # load data frame
    load_loc = 'data/exps/%s/binned/%s.csv' % (exp_name, name)
    print('load data')
    # temporarily parse the dates if they're in the data frame
    df = pd.read_csv(load_loc, parse_dates=[
                     'date_%s' % code for code in all_offr_codes('b3')])

    # remove date columns
    print('Removing date columns')
    df = remove_dates(df)
    # also temporarily extract all reference columns and the original offer columns as well
    # For this experiment, we are not normalizing offers to have 0 mean and 1 standard deviation
    # grab offer codes
    # TODO: The normalized offer problem can be solved creating another set of reference
    # TODO: columns after difference normalization but BEFORE mean and variance normalization
    offr_codes = all_offr_codes('b3')
    # concat with offer name
    offr_cols = ['offr_%s' % code for code in offr_codes]
    # concat with ref to form ref offers
    ref_cols = ['ref_%s' % col for col in offr_cols]
    # concat both lists of columns
    extract_cols = offr_cols + ref_cols
    # delete unnecessary variables
    del offr_cols
    del ref_cols

    print('Removing columns that will not be normalized temporarily')
    # create a dictionary to store the columns in temporarily
    # and remove each from the data frame
    temp_dict = {}
    for col in extract_cols:
        if col in extract_cols:
            temp_dict[col] = df[col].copy()
            df.drop(columns=col, inplace=True)
        else:
            temp_dict[col] = None

    # for debugging purposes, check which columns have been removed from
    # the data frame
    for col in temp_dict.keys():
        print('Removed: %s' % col)

    # create two lists of columns in df -- the first consisting of all columns with
    # na values, the second consisting of all complete columns
    inc_cols, full_cols = get_inc_full_cols(df)

    if name == 'train' or name == 'toy':
        # initialize a list to track dropped columns
        dropped_cols = []
        # initialize lists to track mean, std, and colname for
        # columns that aren't dropped
        mean_list = []
        std_list = []
        name_list = []

        # iterate over each incomplete column
        print('Normalizing mean,std for incomplete columns')
        for colname in inc_cols:
            # extract the column
            col = df[colname]
            # grab the mean and standard deviation
            colmean = col.mean(skipna=True)
            colstd = col.std(skipna=True)
            # find the indices where the column is defined
            full_inds = col[~col.isna()].index
            # check whether the standard deviation is 0
            # if so, remove the column from the data frame
            if colstd == 0:
                df.drop(columns=colname, inplace=True)
                print('Dropping %s due to 0 deviation' % colname)
                sys.stdout.flush()
                # add dropped column to the list of dropped columns
                dropped_cols.append(colname)
            else:
                # otherwise subtract the mean and divide by the standard deviation
                # for all defined entries in the column
                df.loc[full_inds, colname] = (
                    df.loc[full_inds, colname] - colmean) / colstd
                # add the mean, std, and colname to their respective accumulators
                mean_list.append(colmean)
                std_list.append(colstd)
                name_list.append(colname)
                # check whether normalization has  made any previously observed values
                # na
                na_flag = df.loc[full_inds, colname].isna().any()
                # if so, raise a ValueError
                if na_flag:
                    raise ValueError('Normalizing data generated na values')

        # moving on to complete columns
        print('Normalizing completely observed columns')
        # convert the numpy list of columns to an index series
        full_cols = pd.Index(full_cols)
        # grab the mean and standard deviation for each complete column
        mean = df[full_cols].mean()
        std = df[full_cols].std()
        # find column names with 0 deviation
        std_zeros = std[std == 0].index
        # drop these columns from the data frame, mean series, std series,
        # and full columns index
        df.drop(columns=std_zeros, inplace=True)
        mean.drop(index=std_zeros, inplace=True)
        std.drop(index=std_zeros, inplace=True)
        # does not have inplace keyword for index
        full_cols = full_cols.drop(std_zeros)
        # add these columns to the list of dropped cols
        dropped_cols = dropped_cols + list(std_zeros.values)
        # update the values of the columns in the data frame
        df[full_cols] = (df[full_cols] - mean)/std

        # check wether any of the previously full columns now have na values
        na_flag = df.loc[full_cols].isna().any().any()
        # throw error if na values generated
        if na_flag:
            raise ValueError('Normalizing data generated na values')
        # compose mean and std into data frame with 2 columns (mean, std)
        # where each index indicates the corresponding column
        if name == 'train':
            print('Save intermediate outputs for application to test set')
            # pickle norm df for complete columns
            norm_df = pd.DataFrame({'mean': mean, 'std': std})
            norm_df.index.name = 'cols'
            norm_df.to_csv('data/exps/%s/norm.csv' % exp_name)

            # compile additional normalized data in a dictionary
            # meanlist: list of means for columns with np.NaN values
            # stdlist: list of standard deviations for columns with np.NaN values
            # namelist: list of column names for columns with np.NaN values
            # droppedlist: list of column names for columns where the standard dev
            # was equal to 0, forcing them to be removed from the dataframe
            output_dict = {}
            output_dict['meanlist'] = mean_list
            output_dict['stdlist'] = std_list
            output_dict['namelist'] = name_list
            output_dict['droppedlist'] = dropped_cols
            # open a pickle and dump output dictionary into it
            norm_pick = open('data/exps/%s/norm.pickle' %
                             exp_name, 'wb')
            pickle.dump(output_dict, norm_pick)
            norm_pick.close()

    else:
        print('Loading outputs from training normalization')
        # load norm df from training corpus
        norm_df = pd.read_csv('data/exps/%s/norm.csv' % exp_name)
        norm_df.set_index('cols', drop=True, inplace=True)

        # load additional mean,std data and dropped columns from training corpus pickle
        f = open("data/exps/%s/norm.pickle" % exp_name, "rb")
        pic_dic = pickle.load(f)
        f.close()
        # extract relevant lists from dictionary
        mean_list = pic_dic['meanlist']
        std_list = pic_dic['stdlist']
        name_list = pic_dic['namelist']
        dropped_cols = pic_dic['droppedlist']
        # if the length of dropped_cols is greater than 0, drop all columns it contains
        # from the test df
        if len(dropped_cols) > 0:
            df.drop(columns=dropped_cols, inplace=True)

        # grab mean and standard deviation series from norm_df
        # for the full columns
        print('Normalizing fully observed columns')
        mean_ser = norm_df['mean']
        std_ser = norm_df['std']
        # get the index of full columns
        full_cols = mean_ser.index
        # update the values of all full columns
        df.loc[full_cols] = (df[full_cols] - mean_ser) / std_ser
        # check wether any of the previously full columns now have na values
        na_flag = df.loc[full_cols].isna().any().any()
        # throw error if na values generated
        if na_flag:
            raise ValueError('Normalizing data generated na values')
        # zip list of names of incomplete columns, their means, and standard deviations
        # and iterate over it
        print('Normalizing partially observed columns')
        for curr_name, curr_mean, curr_std in zip(mean_list, std_list, name_list):
            curr_col = df[curr_name]
            filled_inds = curr_col[~curr_col.isna()].index
            # update filled inds in curr_col
            df.loc[filled_inds, curr_name] = (
                df.loc[filled_inds, curr_name] - curr_mean) / curr_std
            # check whether any previously filled values have been made na
            na_flag = df.loc[filled_inds, curr_name].isna().any()
            # if so, raise a ValueError
            if na_flag:
                raise ValueError('Normalizing data generated na values')
    # add all removed columns back into the data frame
    print('Adding temporarily dropped columns back to the data frame')
    for col_name, col_ser in temp_dict.items():
        if col_ser is not None:
            df[col_name] = col_ser

    # save the current data frame
    print('Exporting results')
    df.to_csv('data/exps/%s/normed/%s.csv' %
              (exp_name, name), index_label=False)


if __name__ == '__main__':
    main()
