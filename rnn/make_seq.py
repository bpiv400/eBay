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


def get_colnames(columns, const=False, ref=False, b3=False):
    '''
    Extracts the columns in df that correspond to
    constant features of the thread, the offer generated features of the thread,
    or reference columns

    Args:
        columns: String Index or list of strings giving columns in df
        const: a boolean giving whether constant column names should be extracted
        Default is False
        ref: a boolean giving whether reference columns should be extracted
        b3: boolean for whether model we are training will eventually be used to
        predict b3

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
        elif match_offr and not const:
            # add the column to the list if it is a reference column
            # and we're searching for reference columns
            if ref and match_ref:
                out_cols.append(col)
                print('%s treated as ref column' % col)
                sys.stdout.flush()
            # or add the column to the list if it is not a reference
            # column and we're not searching for reference columns
            if not ref and not match_ref:
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
    if not const and not ref:
        # offr_cols should contain only columns ultimately used as sequence inputs
        # therefore we remove all features corresponding to b3
        # and all features corresponding to s2 IF b3 = False
        drop_feats = feats_from_str('_b3$', out_cols)
        if not b3:
            drop_feats = drop_feats + feats_from_str('_s2$', out_cols)
        # define a closure to generate a filter function that returns true
        # only when the column name is not contained in the list of
        # columns we are dropping

        def make_col_filter(drop_feats):
            def col_filter(x):
                return x not in drop_feats
            return col_filter

        # call closure to generate function
        col_filter = make_col_filter(drop_feats)
        # apply filter to out out_cols
        out_cols = filter(col_filter, out_cols)

    return out_cols


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


def seq_lists_legit(out, concat=False, sep=False, b3=False):
    '''
    Ensures the list of sequence features have appropriate relative
    lengths. Specifically, if sep is true, the outer list should
    have length 2, and the first inner list
    should have one fewer element than the second list

    Additionally, all inner elements of both lists should have the
    same number of features UNLESS at some point we develop some
    features for the one of the players that we do not have for the
    other player

    If concat is true, then the first element should have half as
    many features as the remaining the elements, which should all
    have the same number of elements

    Throws an error if anything is wrong

    Can be enhanced to check tags if anything appears wrong as well
    SHOULD BE ENHANCED TO CHECK TAGS
    '''
    # seller feature tag
    slr_tag = r'_s[0-2]$'
    # buyer feature tag
    byr_tag = r'_b[0-2]$'

    if sep:
        # ensure the otuer lister has two elements
        if len(out) != 2:
            raise ValueError(
                'Buyer and seller threads not separated correctly')
        num_byr_seqs = len(out[0])
        num_slr_seqs = len(out[1])
        # ensure the inner list includes one more buyer thread than seller thread
        if num_byr_seqs - num_slr_seqs != 1:
            raise ValueError(
                'Buyer and seller threads not separated correctly')
        # ensure that each buyer and seller thread contain the same number of elements
        # using list comprehension to extract a length for each inner list from
        # the two outer lemeents
        byr_lens = [len(byr_seq) for byr_seq in out[0]]
        slr_lens = [len(slr_seq) for slr_seq in out[1]]
        # ensure that all strings in each byr_seq list contain a byr code
        # by counting the number of strings that match the byr_tag pattern
        matching_byr_lens = [len(feats_from_str(byr_tag, byr_feats))
                             for byr_feats in out[0]]
        # do the same for seller sequences
        matching_slr_lens = [len(feats_from_str(byr_tag, slr_feats))
                             for slr_feats in out[1]]
        # similarly ensure that the seller features contain no byr tags
        byr_tags_slr_seqs = functools.reduce(
            lambda feats, acc: len(feats_from_str(byr_tag, feats)), out[1], 0)
        # do the same for byr threads and the seller tag
        slr_tags_byr_seqs = functools.reduce(
            lambda feats, acc: len(feats_from_str(slr_tag, feats)), out[0], 0)
        # ensure searches for slr tags in byr features and vice versa
        # encountered no matches
        if byr_tags_slr_seqs != slr_tags_byr_seqs or slr_tags_byr_seqs != 0:
            raise ValueError('Seller tag detected in byr feats or vice versa')
        # set baseline length arbitrarily
        init_len = byr_lens[0]
        # iterate over concatented list in search of difference in length or
        # number of tag matches
        for curr_len, curr_matching_len in zip(byr_lens + slr_lens, matching_byr_lens + matching_slr_lens):
            if init_len != curr_len or init_len != curr_matching_len:
                raise ValueError(
                    'Buyer and seller threads not separated correctly')
    elif concat:
        # if concatented, extrac tthe length of the first element
        init_len = len(out[0])
        # find the theoretical length of the next element
        post_len = init_len * 2
        # iterate over all elements except the first
        for i in range(1, len(out)):
            # ensure that each has length equal to double the length of the first element
            curr_list_len = len(out[i])
            if curr_list_len != post_len:
                raise ValueError(
                    'Buyer and seller threads not concatenated correctly')
            curr_list = out[i]
            # ensure that that the list only contains all necessary byr_matches and
            # no seller matches if the index is even and vice versa if odd
            byr_matches = len(feats_from_str(byr_tag, curr_list))
            slr_matches = len(feats_from_str(slr_tag, curr_list))
            # set expectations depending on whether the current set of features
            # should correspond to a buyer or a seller
            if i % 2 == 0:
                exp_byr = post_len
                exp_slr = 0
            else:
                exp_byr = 0
                exp_slr = post_len
            if exp_byr != byr_matches or slr_matches != exp_slr:
                raise ValueError('Buyer and seller thread codes not concatenated correctly' +
                                 'Some buyer threads with sellers or vice versa')
        # ensure the first buyer thread is fully composed of buyer feats
        # and has no seller features, not checked in loop
        byr_matches = len(feats_from_str(byr_tag, out[0]))
        slr_matches = len(feats_from_str(slr_tag, out[0]))
        if byr_matches != init_len or slr_matches != 0:
            raise ValueError('Buyer and seller thread codes not concatenated correctly' +
                             'Some buyer threads with sellers or vice versa')
    else:
        # if not concatenated or separated, ensure that each element of the list
        # has the same length
        init_len = len(out[0])
        for i in range(len(out)):
            curr_list = out[i]
            if len(curr_list) != init_len:
                raise ValueError(
                    'Buyer and seller threads not processed correctly')
            byr_matches = len(feats_from_str(byr_tag, curr_list))
            slr_matches = len(feats_from_str(slr_tag, curr_list))
            # set expectations depending on whether the current set of features
            # should correspond to a buyer or a seller
            if i % 2 == 0:
                exp_byr = init_len
                exp_slr = 0
            else:
                exp_byr = 0
                exp_slr = init_len
            if exp_byr != byr_matches or slr_matches != exp_slr:
                raise ValueError(
                    'Buyer and seller threads not processed correctly' +
                    '. Some buyer threads in seller feats or vice versa')
    pass


def get_seq_lists(offr_cols, concat=False, sep=False, b3=False):
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

    # If b3, should contain b0 -> s2
    # Otherwise, should contain b0 -> s1

    Return:
        If sep = False, list containing string lists
        If sep = True, list containing two lists, where each contains a set
        of string lists
    '''
    #! ERROR  CHECKING BASED ON THE FACT THAT WE ARE REMOVING b3 for the
    #! timebeing
    b3_matches = feats_from_str('_b3$', offr_cols)
    s2_matches = feats_from_str('_s2$', offr_cols)
    if (len(b3_matches) > 0 or len(s2_matches) > 0) and not b3:
        raise ValueError('b3 has not been totally removed')
    #! ##############################################################
    # more error checking for ref columns
    ref_matches = feats_from_str('^ref_', offr_cols)
    if len(ref_matches) > 0:
        raise ValueError('ref columns remain')
    # intentionally excluding b3
    # grab all offer codes
    if b3:
        offr_codes = all_offr_codes('s2')
        diff = 0
    else:
        offr_codes = all_offr_codes('b2')
        diff = 1

    # separate them into buyer and seller offer codes
    byr_codes = feats_from_str('b', offr_codes)
    slr_codes = feats_from_str('s', offr_codes)
    # additional error checking
    act_diff = len(byr_codes) - len(slr_codes)
    if diff != act_diff:
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
        slr_feats = feats_from_str(slr_code, offr_cols)
        slr_seqs.append(slr_feats)
    # if the output should be concatenated
    if concat:
        # first append just the first buyer sequence since there isn't a corresponding
        # preceeding seller offer
        out.append(byr_seqs[0])
        # zip the remainder of the byr_features and the first two elements of the seller
        # features
        for byr_feats, slr_feats in zip(byr_seqs[1:], slr_seqs):
            # append the concatenated list of features from both
            out.append(byr_feats + slr_feats)
    # if byrs and slrs are supposed to be separated into separate np.ararys for input through separate
    # matrices, return a list with two elements where the first is the list of byr feature lists
    # and the second is the list of slr feature lists
    elif sep:
        out = [byr_feats, slr_feats]
    # otherwise return all the buyer and seller features together in an ordered list
    else:
        for byr_feats, slr_feats in zip(byr_seqs, slr_seqs):
            out.append(byr_feats)
            out.append(slr_feats)
    # ensure that the seq_lists are appropriate relative length
    seq_lists_legit(out, concat=concat, sep=sep, b3=b3)
    return out


def get_last_code(length):
    '''
    Maps an integer giving the length of a sequence
    to the code for the last turn in the sequence
    '''
    if length % 2 == 0:
        turn_type = 's'
        adj_len = length - 1
        turn_num = math.floor(adj_len / 2)
    else:
        turn_type = 'b'
        turn_num = math.floor(length / 2)
    turn_code = '%s%d' % (turn_type, turn_num)
    return turn_code


def fix_odd_seqs(df, max_length):
    '''
    When inputs are being concatenated or we are using separate input paths
    for buyer and seller threads, the thread lengths need to be adjusted to
    reflect the fact that there are only 3 steps in RNN instead of 6

    Args:
        df: pd.DataFrame containing only features related to sequence input, ie
        offer threads only for offers b0 -> s1
        sep: boolean for whether seller and buyer offers are being input separately
    '''
    feats = df.columns
    ###############################
    #! b3 ERROR CHECKING
    #! SHOULD ALWAYS BE REMOVED BEFORE
    b3_matches = feats_from_str('_b3$', feats)
    if len(b3_matches) > 0 or max_length >= 7:
        raise ValueError('b3 has not been totally removed')
    #################################
    length = df['length'].copy()
    # subtract one from odd length sequences
    df['length'] = pd.Series((length - (length % 2)), index=df.index)
    # grab min length for error checking
    min_length = df['length'].min()
    if min_length < 2:
        raise ValueError(
            'min length should never be less than 2 at this point')
    # generate list of all remaining offer codes
    all_offrs = all_offr_codes(get_last_code(max_length))
    # extract updated length series
    length = df['length'].copy()
    for i in range(min_length, max_length + 1):
        # get  indices of rows where where the current length is the max length
        curr_inds = length[length == i].index
        # get all observed offer codes
        observed_offr_codes = all_offr_codes(get_last_code(i))
        # generate unobserved offer codes, ie those in all_offrs
        # not in observed_offr_codes
        unobserved_offr_codes = [
            code for code in all_offrs if code not in observed_offr_codes]
        # for each unobserved_offr_code generate a list of matching columns
        cols_for_code_list = [feats_from_str(
            '_%s$' % code, feats) for code in unobserved_offr_codes]
        # flatten list of lists
        flat_cols = [
            col for curr_code_list in cols_for_code_list for col in curr_code_list]
        # set extracted indices for each matching column to 0
        df.loc[curr_inds, flat_cols] = 0
    # after resetting all values to appear as though odd length sequences did not observe the
    # final turn, divide lenght by 2 to reflect actual length in rnn processing
    df['length'] = df['length'] / 2
    pass


def get_seq_vals(df, seq_lists):
    '''
    Assumes ghost features have been added
    '''
    max_len = df['length'].max()
    # get the size of one sequence input
    num_seq_feats = seq_lists[0]
    vals = np.empty((max_len, len(df.index), num_seq_feats))
    # initialize sequence index counter
    seq_ind = 0
    # iterate over lists of features contained in seq_lists
    for curr_feats in seq_lists:
        # final debugging check
        if len(curr_feats) != num_seq_feats:
            raise ValueError('unexpected number of features in sequence')
        # extract columns from the data.frame as 2d numpy array
        # of dim len(df) x num_seq_feats
        curr_vals = df[curr_feats].values
        # nan error checking
        if np.any(np.isnan(curr_vals)):
            raise ValueError('na value detected')
        # insert curr_val matrix at seq_ind in the 3d ndarray
        vals[seq_ind, :, :] = curr_vals
        # increment ind counter
        seq_ind = seq_ind + 1
    return vals


def get_ghost_names(names):
    '''
    Get names of ghosts features corresponding to list of features...
    for a feature 'x_[bs][012]' the ghost feature is 'x_ghost'

    Args:
        names: string list

    Returns: String list of ghost features
    '''
    # generate sequence code for regular expression matching
    seq_code = r'_[bs][0-3]$'
    # remove seq_code and replace with _ghost
    names = [re.sub(seq_code, '_ghost', feat_name) for feat_name in names]
    return names


def ghost_feats(seq_lists, df, concat=False, sep=False):
    '''
    If the experiment uses separated buyer and seller inputs or
    concatenated inputs, add a set of ghost features (where all values are 0)
    to the data frame and the appropriate location in seq_lists to account for
    the fact that the first buyer offer does not have a preceeding seller offer
    '''
    if sep:
        # extract list where the ghost features should be added
        # if separated, this is the list consisting of seller sequences
        underfull_seqs = seq_lists[1]
        ghost_names = get_ghost_names(underfull_seqs[0])
        # add list of ghost names as the element in the list of seller sequences
        underfull_seqs = [ghost_names] + underfull_seqs
        # replace the original list of seller seqs with this new list
        seq_lists[1] = underfull_seqs
    elif concat:
        # extract features contained in the second sequence entry
        # corresponding to the first full entry
        byr_and_slr_feats = seq_lists[1]
        # find all features in this list that have a seller code
        slr_feats = feats_from_str(r'_s[012]$', byr_and_slr_feats)
        # get list of all features contained only in slr_feats
        ghost_names = get_ghost_names(slr_feats)
        # pre-pend these features onto the list of features for the first
        # sequence entry
        seq_lists[0] = ghost_names + seq_lists[0]
    # add ghost names to df
    for curr_name in ghost_names:
        df[curr_name] = pd.Series(0, index=df.index)
    # return sequence list
    return seq_lists


def extract_cols(df, cols=[], is_const=False, sep=False, concat=False, b3=False):
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
        b3: boolean whether offr_b3 will be predicted by the final model

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
        # fill na values with 0's (padding)
        df.fillna(value=0, inplace=True)
        # add length to the data frame
        curr_cols['length'] = df['length']
        # grab the max length
        max_length = curr_cols['length'].max()
        # extract lists of sequence features
        seq_lists = get_seq_lists(cols, concat=concat, sep=sep, b3=b3)
        # if we are concatenating or separating, change thread lengths to reflect
        # how long the threads will be when we input them to the model
        # and hide the final offer features from odd length threads since
        # these cannot be used
        if concat or sep:
            fix_odd_seqs(curr_cols, max_length)
            # additionally add ghost features to data frame and seq_lists
            seq_lists = ghost_feats(seq_lists, df, concat=concat, sep=sep)
        # sort by sequence length in descending order
        df.sort_values(by='length', inplace=True, ascending=False)
        # extract values using seq_lists
        if sep:
            # extract value ndarray for buyers
            byr_vals = get_seq_vals(curr_cols, seq_lists[0])
            # extract value ndarray for sellers
            slr_vals = get_seq_vals(curr_cols, seq_lists[1])
            vals = (byr_vals, slr_vals)
        vals = get_seq_vals(curr_cols, seq_lists)
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
    f = open(path, "rb")
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
    # gives whether the last offer b3 should be kept in the data_frame
    parser.add_argument('--b3', '-b3', action='store_true')

    # parse args
    args = parser.parse_args()
    name = args.name
    exp_name = args.exp
    data_name = args.data
    sep = args.separate
    concat = args.concat
    b3 = args.b3

    # quick argument error checking
    # b3 cannot be active at the same time as sep or concat
    if b3 and (sep or concat):
        raise ValueError('we cannot keep b3, if we are inputting' +
                         'seller and buyer offers separately or if we are concatenating seller' +
                         'and buyer offers')
    # sep and concat cannot be active at the same time
    if sep and concat:
        raise ValueError('We cannot separate AND concatenate buyer and seller' +
                         'offers simultaneously')

    # load data
    load_loc = 'data/exps/%s/normed/%s.csv' % (data_name, name)
    df = pd.read_csv(load_loc)
    # drop unnamed columns if they exist
    drop_unnamed(df)
    #! NOTE: FOR THE TIME BEING DROP b3 offers since the seller cannot
    #! respond and we are presently only training models to predict
    #! seller responses to buyer offers
    #! CAVEAT: predictions made for b3 may help the model generalize for
    #! its earlier seller predictions--therefore, in certain experiments,
    #! we may choose to keep it
    if not b3:
        drop_b3(df)

    # load column name lists if currently processing test data
    if name == 'test':
        const_cols, offr_cols = get_training_features(exp_name)
        # for ref columns, indescriminately grab ALL reference columns
        ref_cols = get_colnames(df.columns, const=False, ref=True, b3=b3)
    else:
        # generate column name lists if the current data corresponds to training or toy data
        const_cols = get_colnames(df.columns, const=True, ref=False, b3=b3)
        # offr_cols should only generate exactly those columns that will be used as input to the
        offr_cols = get_colnames(df.columns, const=False, ref=False, b3=b3)
        # for ref columns, indescriminately grab ALL reference columns
        ref_cols = get_colnames(df.columns, const=False, ref=True, b3=b3)
        feat_dict = {}
        feat_dict['offr_feats'] = offr_cols
        feat_dict['const_feats'] = const_cols

    # extract constant columns and corresponding data prepped for torch.rnn/lstm
    const_vals = extract_cols(df, cols=const_cols, is_const=True)
    # extracts the sequence values numpy arary
    offr_vals = extract_cols(
        df, cols=offr_cols, is_const=False, sep=sep, concat=concat, b3=b3)
    #


if __name__ == '__main__':
    main()
