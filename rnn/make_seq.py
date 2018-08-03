import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import re
import argparse
import math
import gc
import sys
import os
import functools


def get_colnames(columns, const=False, ref=False):
    '''
    Extracts the columns in df that correspond to
    constant features of the thread, the offer generated features of the thread,
    or reference columns

    Args:
        columns: String Index or list of strings giving columns in df
        const: a boolean giving whether constant column names should be extracted
        Default is False
        ref: a boolean giving whether reference columns should be extracted


    Returns:
        A list of column names
    '''
    # define general offer code regular expression
    offer_code_re = r'_[bs][0123]$'
    ref_re = r'^ref_offr_[bs][0123]$'
    # intialize list of all constant columns
    out_cols = []
    # iterate over each column
    for col in columns:
        # determine whether the current offer contains
        # an offer code
        match_offr = re.search(offer_code_re, col)
        # determine whether the current column is a reference column
        match_ref = re.search(ref_re, col)

        # add it to the list of constant columns if
        # its not flagged as having an offer code and
        # the function is collecting constant columns
        if not match_offr and const:
            out_cols.append(col)
            # error checking
            print('%s treated as constant' % col)
            sys.stdout.flush()
        # otherwise if the current column is identified as having an offer
        # code and the function isn't seraching for constant columns
        elif match and not const:
            # add the column to the list if it is a reference column
            # and we're searching for reference columns
            if ref and ref_re:
                out_cols.append(col)
                print('%s treated as ref column' % col)
                sys.stdout.flush()
            # or add the column to the list if it is not a reference
            # column and we're not searching for reference columns
            if not ref and not ref_re:
                out_cols.append(col)
                print('%s treated as offer generated' % col)
                sys.stdout.flush()
            # error checking
    # check whether length is contained in the output list
    # if so, remove it
    if 'length' in out_cols:
        out_cols.remove('length')
    if 'unique_thread_id' in out_cols:
        out_cols.remove('unique_thread_id')

    return const_cols


def pickle_obj(obj, path, filename):
    '''
    Pickles an arbitrary object to an arbitrary file

    Args:
        obj: any object that can be serialized
        path: string giving the relative path to the directory where the object
        be pickled
        filename: string giving the desired name of the file, this should not include
        the extension or the path to the directory

    Returns:
        Nothing
    '''
    # define path to pickle file
    pickle_loc = '%s/%s.pickle' % (path, filename)
    # tracking
    print('Pickling to %s' % pickle_loc)
    # open file, write in raw binary
    obj_pick = open(pickle_loc, 'wb')
    # dump the object
    pickle.dump(obj, obj_pick)
    # close the pickle
    obj_pick.close()
    print('Done pickling to %s' % pickle_loc)
    pass


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


def feats_from_str(refstr, offr_cols):
    '''
    Gets all features in a list of feature names
    which match the code given by str

    Expects code to come at the end of each feature
    name

    Args:
        str: a string to be matches
        offr_cols: list of strings to search

    Returns:
        List of strings extracted from offr_cols
    '''
    matches = [re.search(refstr, col) for col in offr_cols]
    matches = [match.string for match in matches if match]
    return matches


def get_seq_lists(offr_cols, concat=False, sep=False):
    '''
    Extracts list where each entry gives a list of feature names for
    each entry in the sequence. If sep = True, then the list contains two
    elements, the first giving a list of sequence feature lists for buyer
    offers and the second giving a list of sequence feature lists for
    the seller offers

    Args:
        offr_cols: list of strings corresponding to offer columns
        concat: boolean giving whether buyer and seller offer features
        are being concatenated
        sep: boolean giving whether buyer and seller features are
        being input separately downstream

    Return:
        If sep = False, list containing string lists
        If sep = True, list containing two lists, where each contains a set
        of string lists
    '''
    #! ERROR  CHECKING BASED ON THE FACT THAT WE ARE REMOVING b3 for the
    #! timebeing
    b3_matches = feats_from_str('_b3$', offr_cols)
    if len(b3_matches) > 0:
        raise ValueError('b3 has not been totally removed')
    #! ##############################################################
    # more error checking for ref columns
    ref_matches = feats_from_str('^ref_', offr_cols)
    if len(ref_matches) > 0:
        raise ValueError('ref columns remain')
    # intentionally excluding b3
    # grab all offer codes
    offr_codes = all_offr_codes('s2')
    # separate them into buyer and seller offer codes
    byr_codes = feats_from_str('b', offr_codes)
    slr_codes = feats_from_str('s', offr_codes)
    # additional error checking
    if len(byr_codes) != len(slr_codes):
        raise ValueError('There should be one seller offer in response to every buyer' +
                         'offer, no more, no less')
    # initialize list for seller sequences and buyer sequences
    slr_seqs = []
    byr_seqs = []
    # initialize output list
    out = []
    # zip codes into tuple and iterate over them
    for byr_code in byr_codes:
        # add end of line delimiter to prevent accidental matches
        byr_code = '%s$' % byr_code
        # grab all columns containing the buyer code
        byr_feats = feats_from_str(byr_code, offr_cols)
        # append to byr_seqs list
        byr_seqs.append(byr_feats)
    # stop trying to be
    for slr_code in slr_codes:
        slr_code = '%s$' % slr_code
        # grab all columns containing the seller code
        slr_feats = feats_from_str(slr_feats, offr_cols)
        slr_seqs.append(slr_feats)
    # if the output should be concatenated
    if concat:
        # first append just the first buyer sequence since there isn't a corresponding
        # preceeding seller offer
        out.append(byr_seqs[0])
        # zip the remainder of the byr_features and the first two elements of the seller
        # features
        for byr_feats, slr_feats in zip(byr_seqs[1:], slr_seqs[:len(slr_seqs - 1)]):
            # append the concatenated list of features from both
            out.append(byr_feats + slr_feats)
        return out
    # if byrs and slrs are supposed to be separated into separate np.ararys for input through separate
    # matrices, return a list with two elements where the first is the list of byr feature lists
    # and the second is the list of slr feature lists
    elif sep:
        return [byr_feats, slr_feats]
    # otherwise return all the buyer and seller features together in an ordered list
    else:
        for byr_feats, slr_feats in zip(byr_seqs, slr_seqs):
            out.append(byr_feats)
            out.append(slr_feats)
        return out


def extract_cols(df, cols=[], is_const=False, sep=False, concat=False):
    '''
    Extracts features from the data frame, converts the data these
    contain to a numpy.ndarray with size = (1, num_samp, num_feats), and
    deletes the columns in the original pd.DataFrame

    Args:
        df: a pandas.DataFrame containing thread data
        cols: a list of column names extracted from the
        training data frame. Default is an empty list
        is_const: boolean giving whether the current columns being grabbed are
        constant listing features
        concat:  boolean giving whether the model is using concatenated
        seller and buyer features as inputs
        sep: boolean giving whether the model is separating the buyer and
        seller offers as two separate sources of input

    Returns:
        columns formatted as np.array with appropriate dimensionality
    '''
    # extract relevant columns and drop them from data frame after
    # extraction
    curr_cols = df[cols].copy()
    gc.collect()
    df.drop(columns=cols, inplace=True)
    # processing for constant column features
    # should return (1, num_samp, num_feats) np.array
    if is_const:
        # extract the associated values as numpy array of dim (samples, hidden_size)
        vals = curr_cols.values
        # add an empty dimension as the first in the numpy array
        vals = np.expand_dims(vals, 0)
        # finally remove constant features from data frame
    else:
        # get length of each sequence
        lengths = df['length'].values
        # find longest sequence
        max_length = np.amax(lengths)
        # fill na values with 0's (padding)
        df.fillna(value=0, inplace=True)
        seq_feats =
        max_length
    return vals


def drop_unnamed(df):
    '''
    Drop unnamed columns from df--these correspond to useless indices generated accidentally
    by default IO behavior

    Args:
        df: pandas.DataFrame

    Returns:
        pass
    '''
    # iterate over all columns
    for col in df.columns:
        # if the name of the column includes unnamed then drop the column
        if 'Unnamed' in col or 'unnamed' in col:
            df.drop(columns=col, inplace=True)
    pass


def get_training_features(exp_name):
    '''
    Loads the ordered lists of constant and offr generated
    feature names extracted from the training data

    Args:
        exp_name: string giving name of the data/exps directory
        containing relevant data

    Returns:
        Tuple of length two, where each element is a string list.
        The first gives the names of the constant columns,
        the second gives the names of the offer columns
    '''
    # define path names
    path = 'data/exps/%s/prepped/feats.pickle' % exp_name

    # extract from pickles
    feats_dict = unpickle(path)

    # return as tuple
    return feats_dict['const_feats'], feats_dict['offr_feats']


def drop_b3(df):
    '''
    While we are only training a model to predict seller responses to buyer offers and
    not vice versa, drop all b3 features including reference offers and shorten
    the sequence length of the sequences with length 7 to 6

    Args:
        df: pandas.DataFrame containing offer data

    Returns:
        pass
    '''
    # initialize list of b3 offer columns
    b3cols = []
    # iterate over names of all columns
    for col in df.columns:
        # if they contain the b3 offer code, add them to the list
        # will add reference cols
        if '_b3' in col:
            b3cols.append(col)
    # drop all extacted cols
    df.drop(columns=b3cols, inplace=True)
    # extract length series
    len_ser = df['length']

    # get ids of columns with length 7 and set them to length 6
    len7_ids = len_ser[len_ser == 7].index
    # error checking
    if len(len7_ids) == 0:
        print('No length 7 threads')
    # set these to len6 in the original data frame
    df.loc[len7_ids, 'length'] = 6
    pass


def unpickle(path):
    '''
    Extracts an abritrary object from the pickle located at path and returns
    that object

    Args:
        path: string denoting path to pickle

    Returns:
        arbitrary object contained in pickle
    '''
    f = open("data/exps/%s/bins.pickle" % exp_name, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', '-n', action='store',
                        type=str, required=True)
    # gives the name of the current experiment
    parser.add_argument('--exp', '-e', action='store', type=str, required=True)
    # gives the name of the directory where the normalized data with
    # ref cols and lengths can be found
    parser.add_argument('--data', '-d', action='store',
                        type=str, required=True)
    # gives whether seller and buyer offers should be extracted
    # as separate offers or whether they should be concatenated
    parser.add_argument('--concat', '-c', action='store_true')
    # gives whether seller and buyer offers should be extracted into separate
    # numpy arrays for being input into a model downstream separately
    parser.add_argument('--separate', '-s', action='store_true')

    # parse args
    args = parser.parse_args()
    name = args.name
    exp_name = args.exp
    data_name = args.data
    sep = args.separate
    concat = args.concat

    # load data
    load_loc = 'data/exps/%s/normed/%s.csv' % (data_name, name)
    df = pd.read_csv(load_loc)
    # drop unnamed columns if they exist
    drop_unnamed(df)
    #! NOTE: FOR THE TIME BEING DROP b3 offers since the seller cannot
    #! respond and we are presently only training models to predict
    #! seller responses to buyer offers
    drop_b3(df)

    # load column name lists if currently processing test data
    if name == 'test':
        const_cols, offr_cols = get_training_features(exp_name)
    else:
        # generate column name lists if the current data corresponds to training or toy data
        const_cols = get_colnames(df.columns, const=True, ref=False)
        offr_cols = get_colnames(df.columns, const=False, ref=False)
        ref_cols = get_colnames(df.columns, const=False, ref=True)
        feat_dict = {}
        feat_dict['offr_feats'] = offr_cols
        feat_dict['const_feats'] = const_cols

    # extract constant columns and corresponding data prepped for torch.rnn/lstm
    const_vals = extract_cols(df, cols=const_cols, is_const=True)
    offr_vals = extract_cols(
        df, cols=const_cols, is_const=False, sep=sep, concat=concat)
    #


if __name__ == '__main__':
    main()
