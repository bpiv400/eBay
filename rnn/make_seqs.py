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
import copy


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
            # print('%s treated as constant' % col)
            sys.stdout.flush()
        # otherwise if the current column is identified as having an offer
        # code and the function isn't seraching for constant columns
        elif match_offr and not const:
            # add the column to the list if it is a reference column
            # and we're searching for reference columns
            if ref and match_ref:
                out_cols.append(col)
                # print('%s treated as ref column' % col)
                sys.stdout.flush()
            # or add the column to the list if it is not a reference
            # column and we're not searching for reference columns
            if not ref and not match_ref:
                out_cols.append(col)
                # print('%s treated as offer generated' % col)
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
        # debugging
        print('offr feats')
        # define a closure to generate a filter function that returns true
        # only when the column name is not contained in the list of
        # columns we are dropping

        def make_col_filter(drop_feats):
            def col_filter(x):
                return x not in drop_feats
            return col_filter
        # call closure to generate function
        drop_col_filter = make_col_filter(drop_feats)
        # print(drop_col_filter)
        # apply filter to out out_cols
        out_cols = list(filter(drop_col_filter, out_cols))
    if ref:
        out_cols.append('ref_start_price_usd')
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
    sys.stdout.flush()
    # open file, write in raw binary
    obj_pick = open(pickle_loc, 'wb')
    # dump the object
    pickle.dump(obj, obj_pick, protocol=4)
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


def neutralize_feats(feats):
    '''
    Remove offer codes from a list of offer generated feature names
    '''
    feats = copy.copy(feats)
    any_tag = r'_([sb][0-3]|ghost)$'
    feats = [re.sub(any_tag, '', curr_feat) for curr_feat in feats]
    return feats


def same_contents(nested_list):
    '''
    Ensures that element in a nested string list
    contains the same elements
    '''
    # arbitrarily grab the first element from the outer list
    first_el = nested_list[0]
    # iterate over every other element in the outer list
    for curr_el in nested_list[1:]:
        # zip together the current element and the first element and
        # iterate over them in tandem
        for a, b in zip(first_el, curr_el):
            # if the contents dont match at any point, return false
            if a != b:
                return False
    # if we make it out of the loop, all contents must be identical
    return True


def seq_lists_legit(out, concat=False, sep=False, b3=False, ghost=False):
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
    # debugging
    print('input')
    print(out)
    sys.stdout.flush()
    # for user interaction
    if ghost:
        print('Checking legitimacy of ghosted seq_lists')
    else:
        print('Checking legitimacy of ordinary seq_lists')
    # seller feature tag
    slr_tag = r'_s[0-3]$'
    # buyer feature tag
    byr_tag = r'_b[0-3]$'
    # ghost tag
    ghost_tag = r'_ghost$'
    if sep:
        # ensure the otuer lister has two elements
        if len(out) != 2:
            raise ValueError(
                'Buyer and seller threads not separated correctly')
        num_byr_seqs = len(out[0])
        num_slr_seqs = len(out[1])
        # ensure the inner list includes one more buyer thread than seller thread
        if num_byr_seqs - num_slr_seqs != 1 and not ghost:
            raise ValueError(
                'Buyer and seller threads not separated correctly')
        elif num_byr_seqs - num_slr_seqs != 0 and ghost:
            raise ValueError(
                'Buyer and seller threads not separated correctly')
        # strip offr tags from byr and seller lists
        byr_neut = [neutralize_feats(feats) for feats in out[0]]
        slr_neut = [neutralize_feats(feats) for feats in out[1]]
        if not same_contents(byr_neut) or not same_contents(slr_neut):
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
        matching_slr_lens = [len(feats_from_str(slr_tag, slr_feats))
                             for slr_feats in out[1]]
        # if ghost adjustment has been made, the first entry of the seller sequences
        # should contain no seller codes, yet it should contain the same number
        # of buyer codes
        if ghost:
            if matching_slr_lens[0] != 0:
                raise ValueError('Ghost features added incorrectly')
            # if the first seller entry actually contains 0 seller codes,
            # replace the count of seller codes with a count of ghost codes
            # since these should occur in every feature in the first seller
            # sequence entry
            matching_slr_lens[0] = len(feats_from_str(ghost_tag, out[1][0]))
        # similarly ensure that the seller features contain no byr tags
        byr_tags_slr_seqs = functools.reduce(
            lambda acc, feats: len(feats_from_str(byr_tag, feats)) + acc, out[1], 0)
        # do the same for byr threads and the seller tag
        slr_tags_byr_seqs = functools.reduce(
            lambda acc, feats: len(feats_from_str(slr_tag, feats)) + acc, out[0], 0)
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
                print(curr_len)
                print(init_len)
                print(curr_matching_len)
                raise ValueError(
                    'Buyer and seller threads not separated correctly')
    elif concat:
        # if concatented, extrac tthe length of the first element
        init_len = len(out[0])
        # find the theoretical length of the next element
        if not ghost:
            post_len = init_len * 2
        else:
            post_len = init_len  # since the ghost features should equalize the lengths

        # set expectations for the number of seller and buyer features in each of non-initial
        # sequence entries
        exp_byr = post_len / 2
        exp_slr = post_len / 2
        exp_ghost = 0
        # iterate over all elements except the first
        for i in range(1, len(out)):
            # ensure that each has length equal to expected feature length
            curr_list_len = len(out[i])
            if curr_list_len != post_len:
                raise ValueError(
                    'Buyer and seller threads not concatenated correctly')
            curr_list = out[i]
            # ensure that that the list only contains all necessary byr_matches and
            # no seller matches if the index is even and vice versa if odd
            byr_matches = len(feats_from_str(byr_tag, curr_list))
            slr_matches = len(feats_from_str(slr_tag, curr_list))
            ghost_matches = len(feats_from_str(ghost_tag, curr_list))
            if exp_byr != byr_matches or slr_matches != exp_slr or ghost_matches != exp_ghost:
                raise ValueError('Buyer and seller thread codes not concatenated correctly' +
                                 'Some buyer threads with sellers or vice versa')
        # ensure the first buyer thread is fully composed of buyer feats
        # and has no seller features, not checked in loop
        byr_matches = len(feats_from_str(byr_tag, out[0]))
        slr_matches = len(feats_from_str(slr_tag, out[0]))
        ghost_matches = len(feats_from_str(ghost_tag, out[0]))
        if ghost:
            exp_byr = init_len / 2
            exp_ghost = init_len / 2
            exp_slr = 0
        else:
            exp_byr = init_len
            exp_slr = 0
            exp_ghost = 0
        if exp_byr != byr_matches or slr_matches != exp_slr or ghost_matches != exp_ghost:
            raise ValueError('Buyer and seller thread codes not concatenated correctly' +
                             'Some buyer threads with sellers or vice versa')
        # if ghost, check that each element contains the same neutralized features
        # this will need to be changed if byr and seller turns have different features later
        neut_feats = [neutralize_feats(feats) for feats in out]
        if not ghost:
            neut_feats[0] = neut_feats[0] + neut_feats[0]
        if not same_contents(neut_feats):
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
            ghost_matches = len(feats_from_str(ghost_tag, curr_list))
            # set expectations depending on whether the current set of features
            # should correspond to a buyer or a seller
            if i % 2 == 0:
                exp_byr = init_len
                exp_slr = 0
            else:
                exp_byr = 0
                exp_slr = init_len
            exp_ghost = 0
            if exp_byr != byr_matches or slr_matches != exp_slr or ghost_matches != exp_ghost:
                raise ValueError('Buyer and seller thread codes not concatenated correctly' +
                                 'Some buyer threads with sellers or vice versa')
        # debugging
        # print(out)
        # sys.stdout.flush()
        # check that each element of the list has the same contents
        neut_seqs = [neutralize_feats(feats) for feats in out]
        if not same_contents(neut_seqs):
            raise ValueError(
                'Buyer and seller threads not processed correctly')
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
    # debugging
    # print(offr_cols)
    # sys.stdout.flush()
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
    # deubgging
    # print('buyer codes')
    # print(byr_codes)
    # print('slr codes')
    # print(slr_codes)
    # sys.stdout.flush()
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
        byr_reg = '%s$' % byr_code
        # grab all columns containing the buyer code
        byr_feats = feats_from_str(byr_reg, offr_cols)
        byr_feats = neutralize_feats(byr_feats)
        byr_feats = sorted(byr_feats)
        byr_feats = ['%s_%s' % (feat, byr_code) for feat in byr_feats]
        # append to byr_seqs list
        byr_seqs.append(byr_feats)
    # stop trying to be
    for slr_code in slr_codes:
        slr_reg = '%s$' % slr_code
        # grab all columns containing the seller code
        slr_feats = feats_from_str(slr_reg, offr_cols)
        slr_feats = neutralize_feats(slr_feats)
        slr_feats = sorted(slr_feats)
        slr_feats = ['%s_%s' % (feat, slr_code) for feat in slr_feats]
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
        out = [byr_seqs, slr_seqs]
    # otherwise return all the buyer and seller features together in an ordered list
    else:
        if not b3:
            out.append(byr_seqs[0])
            byr_seqs = byr_seqs[1:]
        for byr_feats, slr_feats in zip(byr_seqs, slr_seqs):
            if not b3:
                out.append(slr_feats)
                out.append(byr_feats)
            else:
                out.append(byr_feats)
                out.append(slr_feats)
    # debugging
    # print('seq list output')
    # print(out)
    # sys.stdout.flush()
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


def fix_lengths(df, max_length, sep_concat=False):
    '''
    When inputs are being concatenated or we are using separate input paths
    for buyer and seller threads, the thread lengths need to be adjusted to
    reflect the fact that there are only 3 steps in RNN instead of 6

    Args:
        df: pd.DataFrame containing only features related to sequence input, ie
        offer threads only for offers b0 -> s1
        sep_concat: boolean for whether seller and buyer offers are being input separately
    '''
    # print('is sep_concat: %d' % sep_concat)
    feats = df.columns
    if sep_concat:
        ###############################
        #! b3 ERROR CHECKING
        #! SHOULD ALWAYS BE REMOVED BEFORE
        b3_matches = feats_from_str('_b3$', feats)
        if len(b3_matches) > 0 or max_length >= 7:
            raise ValueError('b3 has not been totally removed')
        #################################

    length = df['length'].copy()
    # subtract one from odd length sequences
    if sep_concat:
        length = pd.Series((length - (length % 2)), index=df.index)
    # subtract one from all lengths
    df['length'] = pd.Series((length - 1), index=df.index)
    # grab min length for error checking
    min_length = df['length'].min()
    if min_length < 1:
        raise ValueError(
            'min length should never be less than 2 at this point')
    # generate list of all remaining offer codes
    all_offrs = all_offr_codes(get_last_code(max_length))
    # extract updated length series
    length = df['length'].copy()
    length_vals = np.unique(df['length'])
    # error checking
    if sep_concat:
        evs = np.any(length_vals % 2 == 0)
        if evs:
            print(length_vals)
            print(evs)
            sys.stdout.flush()
            raise ValueError('Should not have any even lengths')
    # iterate over length values
    for i in length_vals:
        # get  indices of rows where where the current length is the max length
        curr_inds = length[length == i].index
        # get all observed offer codes
        observed_offr_codes = all_offr_codes(get_last_code(i))
        # generate unobserved offer codes, ie those in all_offrs
        # not in observed_offr_codes
        unobserved_offr_codes = [
            code for code in all_offrs if code not in observed_offr_codes]
        # debugging
        if sep_concat:
            true_len = (i + 1) / 2
        else:
            true_len = i
        if not sep_concat:
            print('Codes hidden for org length %d, true length %d:' %
                  (i + 1, true_len))
        else:
            print('Codes hidden for org length %d or %d, true length %d:' %
                  (i + 1, i + 2, true_len))
        print(unobserved_offr_codes)
        sys.stdout.flush()
        # for each unobserved_offr_code generate a list of matching columns
        cols_for_code_list = [feats_from_str(
            '_%s$' % code, feats) for code in unobserved_offr_codes]
        # flatten list of lists
        flat_cols = [
            col for curr_code_list in cols_for_code_list for col in curr_code_list]
        # set extracted indices for each matching column to 0
        df.loc[curr_inds, flat_cols] = 0
    if sep_concat:
        # after resetting all values to appear as though odd length sequences did not observe the
        # final turn, divide lenght by 2 to reflect actual length in rnn processing
        df['length'] = (df['length'] + 1) / 2
    pass


def get_seq_vals(df, seq_lists, is_targ=False, midpoint_ser=None):
    '''
    Assumes ghost features have been added
    '''
    if not is_targ:
        max_len = df['length'].max()
        max_len = np.int(max_len)
    else:
        max_len = len(seq_lists)
        print(max_len)
    # get the size of one sequence input
    num_seq_feats = len(seq_lists[0])
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
        curr_vals = df[curr_feats].copy()
        # nan error checking
        if is_targ:
            # debugging
            print('Targ %d' % seq_ind)
            print(curr_feats)
            sys.stdout.flush()
            curr_vals.fillna(-100, inplace=True)
            curr_vals = curr_vals.squeeze()
            # print('Mean: %.2f' % curr_vals.mean())
            # print('Std: %.2f' % curr_vals.std())
            # print(midpoint_ser)
            sys.stdout.flush()
            curr_vals = midpoint_ser.loc[curr_vals.values]
            curr_vals = curr_vals.to_frame()
            # print(curr_vals)
            # sys.stdout.flush()
        # convert to numpy array
        curr_vals = curr_vals.values
        if np.any(np.isnan(curr_vals)):
            raise ValueError('na value detected')
        # insert curr_val matrix at seq_ind in the 3d ndarray
        vals[seq_ind, :, :] = curr_vals
        # increment ind counter
        seq_ind = seq_ind + 1
    # if targ values, remove the size 1 third dimension
    if is_targ:
        vals = np.squeeze(vals, axis=2)
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
    # processing for constant column features
    # should return (1, num_samp, num_feats) np.array
    if is_const:
        # extract relevant columns and drop them from data frame after
        # extraction
        curr_cols = df[cols].copy()
        gc.collect()
        df.drop(columns=cols, inplace=True)
        # extract the associated values as numpy array of dim (samples, hidden_size)
        vals = curr_cols.values
        # add an empty dimension as the first in the numpy array
        vals = np.expand_dims(vals, 0)
        # finally remove constant features from data frame
    else:
        seq_lists = cols
        # extract values using seq_lists
        if sep:
            # extract value ndarray for buyers
            byr_vals = get_seq_vals(df, seq_lists[0])
            # extract value ndarray for sellers
            slr_vals = get_seq_vals(df, seq_lists[1])
            vals = (byr_vals, slr_vals)
        else:
            vals = get_seq_vals(df, seq_lists)
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
    path = 'data/exps/%s/feats.pickle' % exp_name

    # extract from pickles
    feats_dict = unpickle(path)

    # return as tuple
    return (feats_dict['const_feats'], feats_dict['offr_feats'],
            feats_dict['seq_feats'], feats_dict['targ_feats'])


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


def get_targ_lists(concat=False, sep=False, b3=False, time_mod=False):
    '''
    Extract target columns in order as a list of
    single-item lists

    For concatenation or separation, include only seller
    offers as the targets. Otherwise, include both
    '''
    if b3:
        # grab all target codes through b3
        targ_codes = all_offr_codes('b3')
        # drop b0
        targ_codes.remove('b0')
    else:
        # grab all offer codes through s2
        targ_codes = all_offr_codes('s2')
        # drop b0
        targ_codes.remove('b0')
    if sep or concat:
        # drop all buyer offers
        targ_codes = [code for code in targ_codes if 's' in code]
    # add offer prefix to each
    targ_codes = ['offr_%s' % code for code in targ_codes]
    # give each offr its own list to match seq_lists format
    targ_codes = [[offr] for offr in targ_codes]
    return targ_codes


def get_byr_inds(seq_lists, df, train=True):
    # regex expression for offer codes
    offer_code_re = r'_([bs][0-9])$'
    # initialize list counter
    counter = 0
    # debugging
    # print(seq_lists)
    # iterate over all feature lists in seq_lists
    for feat_list in seq_lists:
        # debugging
        # print(feat_list)
        # sys.stdout.flush()
        # arbitrarily grab the first feature in the list
        arb_feat = feat_list[0]
        # find offer code in the first feature
        code_match = re.search(offer_code_re, arb_feat)
        # extract matching substring
        code_string = code_match.group(1)
        # generate byr_ind for this match
        byr_ind_name = 'byr_ind_%s' % code_string
        # generate series to contain indicator
        byr_ser = pd.Series(np.NaN, index=df.index)
        # find indices where corresponding offer is not nan
        curr_offr_name = 'offr_%s' % code_string
        curr_offr = df[curr_offr_name].copy()
        filled_inds = curr_offr[~curr_offr.isna()].index
        # determine whether the buyer indicator should be activated
        # for the current offer
        ind_val = 1 if (code_string[0] == 'b') else 0
        byr_ser.loc[filled_inds] = ind_val
        # add series back to data frame
        df[byr_ind_name] = byr_ser
        if train:
            # add indicator as the final entry in the current sequence list
            seq_lists[counter].append(byr_ind_name)
        # increment counter
        counter = counter + 1
    if train:
        return seq_lists
    else:
        pass


def get_architecture_mode(exp_name):
    '''
    Return length 2 tuple of separate
    inputs flag and concatenated inputs flag in that order
    '''
    if 'simp' in exp_name:
        return False, False
    elif 'cat' in exp_name:
        return False, True
    elif 'sep' in exp_name:
        return True, False
    else:
        raise ValueError('No architecture mode specified in name')


def get_prep_type(data_name):
    '''
    Extracts the prep type from a data sequence name
    '''
    arch_type = r'_(simp|cat|sep)'
    arch_match = re.search(arch_type, data_name)
    start_arch = arch_match.span(0)[0]
    prep_type = data_name[:start_arch]
    return prep_type

# TODO: AT SOME POINT, CHECK CORRECT


def add_turn_indicators(offr_vals, length_vals):
    # get total number of turns
    max_turns = offr_vals.shape[0]
    # get batch size
    batch_size = offr_vals.shape[1]
    # iterate up to the max number of turns
    for i in range(max_turns):
        # generate a max_turns, batch_size, 1 array of 0's
        curr_ind = np.zeros((max_turns, batch_size, 1))
        # get the current length being operated on
        curr_length = i + 1
        # count the number of threads with at least this length
        count_seqs = np.sum(length_vals >= curr_length)
        # activate indicator for corresponding threads
        curr_ind[i, :count_seqs, 0] = 1
        # append to the last dimension
        offr_vals = np.append(offr_vals, curr_ind, axis=2)
    return offr_vals


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', '-n', action='store',
                        type=str, required=True)
    # gives the name of the current data type
    parser.add_argument('--data', '-d', action='store',
                        type=str, required=True)
    # gives whether the last offer b3 should be kept in the data_frame
    parser.add_argument('--b3', '-b3', action='store_true', default=False)
    # gives whether turn number indicators should be added
    parser.add_argument('--inds', '-i', action='store_true', default=False)
    # parse args
    args = parser.parse_args()
    name = args.name
    data_name = args.data
    b3 = args.b3
    inds = args.inds

    # get architecture mode and set flags appropriately
    sep, concat = get_architecture_mode(data_name)
    prep_type = get_prep_type(data_name)
    # quick argument error checking
    # b3 cannot be active at the same time as sep or concatPrepP
    if b3 and (sep or concat):
        raise ValueError('we cannot keep b3, if we are inputting' +
                         'seller and buyer offers separately or if we are concatenating seller' +
                         'and buyer offers')
    # sep and concat cannot be active at the same time
    if sep and concat:
        raise ValueError('We cannot separate AND concatenate buyer and seller' +
                         'offers simultaneously')

    # load data
    print('Loading data')
    sys.stdout.flush()
    load_loc = 'data/exps/%s/normed/%s.csv' % (prep_type, name)
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
        print('Dropping unnecessary columns')
        sys.stdout.flush()
        drop_b3(df)
    # load midpoints
    print('Preparing midpoints series')
    sys.stdout.flush()
    midpoint_dict = unpickle('data/exps/%s/bins.pickle' % prep_type)
    # extract numpy array of midpoints
    midpoint_list = midpoint_dict['midpoints']
    # ensure rounding
    midpoint_list = np.around(midpoint_list, 2)
    # generate a series from the midpoints where the index is the value of the
    # midpoint and the value of the series is the index of the corresponding midpoint
    # list
    # ensure we append -100 for missing values to both
    midpoint_vals = list(range(len(midpoint_list)))
    midpoint_vals.append(-100)
    midpoint_inds = np.append(midpoint_list, -100)
    midpoint_ser = pd.Series(midpoint_vals, index=midpoint_inds)

    # debugging
    # print('midpoint series')
    # print(midpoint_ser)
    # sys.stdout.flush()

    ################################################
    # NOTE: DEPRECATED
    # Deprecate  maybe?
    # work around since the insert above did not work
    # midpoint_ind = df.index.tolist()
    # idx = len(midpoint_ind) - 1
    # midpoint_ind[idx] = -100
    # midpoint_ser.index = midpoint_ind
    #############################################

    # delete used variables
    del midpoint_list
    del midpoint_dict
    del midpoint_vals
    del midpoint_inds

    # load column name lists if currently processing test data
    if name == 'test':
        print('Loading training feature dictionary')
        sys.stdout.flush()
        const_cols, offr_cols, seq_lists, targ_lists = get_training_features(
            data_name)
        # for ref columns, indescriminately grab ALL reference columns
        ref_cols = get_colnames(df.columns, const=False, ref=True, b3=b3)
        if not sep and not concat:
            get_byr_inds(seq_lists, df, train=False)
    else:
        print('Generating constant/offer cols & sequence lists')
        sys.stdout.flush()
        # generate column name lists if the current data corresponds to training or toy data
        const_cols = get_colnames(df.columns, const=True, ref=False, b3=b3)
        # offr_cols should only generate exactly those columns that will be used as input to the
        offr_cols = get_colnames(df.columns, const=False, ref=False, b3=b3)
        # for ref columns, indescriminately grab ALL reference columns
        ref_cols = get_colnames(df.columns, const=False, ref=True, b3=b3)
        # get target lists in the same format as seq_lists
        # as a list of lists..here however, each list has length 1
        targ_lists = get_targ_lists(concat=concat, sep=sep, b3=b3)
        # grab list of lists where each element corresopnds to a set of features for the
        # sequence entry in the same position
        seq_lists = get_seq_lists(offr_cols, concat=concat, sep=sep, b3=b3)
        # ensure sequence list is valid
        seq_lists_legit(seq_lists, concat=concat, sep=sep, b3=b3, ghost=False)
        # add buyer indicator to each offer if buyer and seller inputs pass through the
        # same weights
        if not sep and not concat:
            seq_lists = get_byr_inds(seq_lists, df, train=True)
            # ensure sequence list is valid
            seq_lists_legit(seq_lists, concat=concat,
                            sep=sep, b3=b3, ghost=False)
        # initialize feature dictionary
        feat_dict = {}
        feat_dict['seq_feats'] = copy.deepcopy(seq_lists)
        feat_dict['offr_feats'] = offr_cols
        feat_dict['const_feats'] = const_cols
        feat_dict['targ_feats'] = copy.deepcopy(targ_lists)
        # pickle feature dictionary
        if name == 'train':
            print('Saving training dictionary')
            sys.stdout.flush()
            feat_path = 'data/exps/%s' % data_name
            feat_name = 'feats'
            pickle_obj(obj=feat_dict, path=feat_path, filename=feat_name)
    if sep or concat:
        print('Getting ghost features')
        sys.stdout.flush()
        # grab sequence list and add ghost features
        seq_lists = ghost_feats(seq_lists, df, sep=sep, concat=concat)
        # ensure the seq list is still valid after adding ghost features
        seq_lists_legit(seq_lists, concat=concat, sep=sep, b3=b3, ghost=True)

    # if we are concatenating or separating, change thread lengths to reflect
    # how long the threads will be when we input them to the model
    # and hide the final offer features from odd length threads since
    # these cannot be used

    # sort by sequence length in descending order
    print('Sorting by length')
    sys.stdout.flush()
    # may sort before fixing lengths because length maps are monotonic increasing
    df.sort_values(by='length', inplace=True, ascending=False)

    # extract target values
    print('Extracting target values')
    sys.stdout.flush()
    print(targ_lists)
    targ_vals = get_seq_vals(
        df, targ_lists, is_targ=True, midpoint_ser=midpoint_ser)

    # get reference data frame and save it
    print('Extracting reference data frame')
    sys.stdout.flush()
    ref_df = df[ref_cols].copy()

    print('Updating thread lengths to reflect rnn architecture')
    # grab the max length
    max_length = df['length'].max()
    sys.stdout.flush()
    fix_lengths(df, max_length, sep_concat=(sep or concat))

    # fill na values with 0's (padding)
    df.fillna(value=0, inplace=True)

    # extract constant columns and corresponding data prepped for torch.rnn/lstm
    print('Extracting constant features')
    sys.stdout.flush()
    const_vals = extract_cols(df, cols=const_cols, is_const=True)
    # extracts the sequence values numpy arary
    print('Extracting sequence features')
    sys.stdout.flush()
    offr_vals = extract_cols(
        df, cols=seq_lists,  is_const=False, sep=sep, concat=concat, b3=b3)
    # add turn number indicators if flag activated
    if inds:
        offr_vals = add_turn_indicators(offr_vals, df['length'].values)
    # store data in dictionary and pickle it
    print('Compiling data dictionary')
    data_dict = {}
    data_dict['const_vals'] = const_vals
    data_dict['ref_vals'] = ref_df
    data_dict['offr_vals'] = offr_vals
    data_dict['target_vals'] = targ_vals
    data_dict['midpoint_ser'] = midpoint_ser
    data_dict['length_vals'] = df['length'].values
    # including for debugging
    data_dict['unique_thread_id'] = df['unique_thread_id'].values
    data_path = 'data/exps/%s' % data_name
    data_dict_name = '%s_data' % name
    sys.stdout.flush()
    print('Pickling data dictionary')
    sys.stdout.flush()
    pickle_obj(obj=data_dict, path=data_path, filename=data_dict_name)
    print('Done preparing for training')


if __name__ == '__main__':
    main()
