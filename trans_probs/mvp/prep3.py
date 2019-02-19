import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse


def get_resp_offr(turn):
    '''
    Description: Determines the name of the response column given the name of the last observed turn
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


def get_all_offrs(turn):
    '''
    Description: Gets the names of all the offer columns in the data set
    including start_price_usd and the response column, returned as a list
    '''
    offer_list = get_observed_offrs(turn)
    resp_offr = get_resp_offr(turn)
    offer_list.append(resp_offr)
    return offer_list


def get_observed_offrs(turn):
    '''
    Description: Generates a list of the offers that have been observed
    thusfar, including the current offer
    '''
    turn_num = int(turn[1])
    turn_type = turn[0]
    offer_set = []
    for i in range(turn_num + 1):
        offer_set.append('offr_b' + str(i))
        if i < turn_num:
            offer_set.append('offr_s' + str(i))
        elif turn_type == 's':
            offer_set.append('offr_s' + str(i))
    offer_set.append('start_price_usd')
    return offer_set


def get_ref_offrs(offr_name):
    '''
    Description: returns a tuple of strings (len 2) of the offer names 
    of the reference offers required to normalize the turn associated with offer name
    Since start_price_usd isn't normalized, this should not be passed as an input.
    Additionally, since the only the start_price_usd precedes offr_b0, offr_b0
     will return only one reference column, instead of a tuple of 2
     --namely start_price_usd
     '''
    # error check for start_price_usd
    if offr_name == 'start_price_usd':
        raise ValueError(
            "ref offers should not be computed for start_price_usd")
    # otherwise remove offr_ substring to isolate turn
    turn = offr_name.replace('offr_', '')
    # use the turn number to get the previous turn
    prev_off = get_prev_offr(turn)
    # if the previous turn is start_price_usd, ie if the current turn is b0,
    # just return the previous offer (tuple len 1)
    if prev_off == 'start_price_usd':
        return prev_off, None
    # otherwise remove offr_ substring from previous offer and again compute
    # the previous offer
    before_prev_turn = prev_off.replace('offr_', '')
    before_prev_offr = get_prev_offr(before_prev_turn)
    return prev_off, before_prev_offr


def norm_by_recent_offers(df, turn):
    # get all offers in the data set, including start and
    # the last offer (response offer) and remove start price
    all_offrs = get_all_offrs(turn)
    all_offrs.remove('start_price_usd')
    # iterate over all remaining offers in the list and
    # compute the tuple of reference offers for each (meaning the
    # two most recent offers)
    ref_offrs = [get_ref_offrs(offr_name) for offr_name in all_offrs]
    # create a data frame containing one column for each previous offer
    normed_df = pd.DataFrame(0.0, index=df.index, columns=all_offrs)
    # create an empty array to populate with ids to drop
    drop_ids = np.zeros(0)
    # iterate over tuple of reference offers and associate offer
    for ref_tup, curr_offr in zip(ref_offrs, all_offrs):
        # from the tuple of reference offers extract previous offer
        # and old offer
        prev_offr = ref_tup[0]
        old_offr = ref_tup[1]
        print('Current_offer: %s' % curr_offr)
        print('Reference Offers: (%s, %s)' % ref_tup)
        # if the old offer is none, meaning curr_offr is b0,
        # just use start_price_usd to normalize offers
        if old_offr is None:
            normed_df[curr_offr] = df[curr_offr] / df['start_price_usd']
        # otherwise compute the normalized value of the offer --
        # the difference between this offer and the last offer normalized by the
        # difference between the old offer and the previous offer
        # this corresponds to a value of 0 when the current offer is the same as the old
        # offer, ie when the player has not compromised ast all and
        # 1 when the difference is the same as the difference between previous offer
        # old offer, corresopnding to the current player accepting the last player's offer
        else:
            normed_df[curr_offr] = (
                df[curr_offr] - df[old_offr]) / (df[prev_offr] - df[old_offr])
        # grab ids of rows with np.nan -- indicates division of 0 by 0
        # this implies thread where the player made an offer of the same value
        # as the other players most recent counter offer (instead of just accepting
        # it)
        # encode these as 1 for the current offer
        nan_ids = normed_df[normed_df[curr_offr].isna()].index.values
        # debugging
        # print(curr_offr)
        # print(nan_ids)
        normed_df.loc[nan_ids, curr_offr] = 1

        # grab id's of rows with -np.inf, np.inf
        inf_ids = normed_df[normed_df[curr_offr] == np.inf].index.values
        ninf_ids = normed_df[normed_df[curr_offr] == -np.inf].index.values
        # combine all into a single np.array
        curr_drop_ids = np.append(ninf_ids, inf_ids)
        # append this np array onto the running arary of ids to drop
        drop_ids = np.append(drop_ids, curr_drop_ids)

    # iterate over every column in the normalized df and replace the columns
    # in the output df with these columns
    # print(df.loc[np.unique(drop_ids), np.append(
    #     normed_df.columns.values, ['start_price_usd', 'unique_thread_id'])])
    for col in normed_df.columns:
        df[col] = normed_df[col]
    # now drop rows where one of the offr columns equals nan, inf, or -inf
    drop_ids = np.unique(drop_ids)
    print('Num dropped: %d' % len(drop_ids))
    # print(df.loc[drop_ids, ['start_price_usd', 'org_offr_b0', 'org_offr_s0']])
    df.drop(index=drop_ids, inplace=True)
    # return the original data frame, now with normalized offer values
    return df


def round_inds(df, round_vals, turn):
    '''
    Description: Create indicators for whether each offer in the data set 
    is exactly equal to some round value (create one such indicator for each 
    offer and round value pair). Additionally, for each offer in the data set, create
    an indicator for whether the offer is close (within 1%) but not equal to any of 
    the round values used to create indicators. Does create indicators for start_price_usd
    since this is grabbed by get_observed_offrs, BUT doesn't create indicators for the response
    offer
    Inputs:
        df: a pandas data frame containing offers
        round_vals: a list of values considered 'round', ie those that people may be 
        likely to converge to (eg [1, 5, 10, 25])
        turn: a string giving the name of the current turn
    Output: a pandas data frame containing indicators described above
    '''
    round_vals = [int(round_val) for round_val in round_vals]
    # get list of all offers in the data set except for the next
    # offer (ie response offer)
    offer_set = get_observed_offrs(turn)

    # for debugging
    # print(offer_set)
    # sample_ind = df[df['unique_thread_id'].isin([96, 167, 206])].index
    # print(sample_ind)
    # iterate over round values
    for offr_name in offer_set:
        # print(offr_name)
        # iterate over all offers the data set except the next offr
        # create series to encode whether the offer in question
        # is near but not directly at a round value
        slack_ser = pd.Series(0, index=df.index)
        # name slack column for current offer
        slack_ser_name = 'slk_%s' % offr_name
        # create empty numpy arrays to track
        # all of the round indices observed so far and all of the
        # slack indices because offers considered 'slack' must
        # not be equal to any of the even values, not just the current one
        # so we track all slack inds and round inds over all iterations
        all_zero_inds = np.ones(0)
        all_nonzero_inds = np.ones(0)
        for curr_val in round_vals:
            # print('val: %s' % str(curr_val))
            # create series for round indicator for current pair of
            # offer and value
            curr_ser = pd.Series(0, index=df.index)
            # give a name to the current round series
            new_feat_name = 'rnd_%d_%s' % (curr_val, offr_name)

            # grab a series of the current offer consisting of rows where the
            # offer in this iteration is nonzero because we must divide by this value
            # to determine slack indices -- which would create an NA headache that would
            # ruin my life for sure
            non_zero_offs = df.loc[df[df[offr_name]
                                      != 0].index, offr_name].copy()
            # debugging
            print('1:')
            print('Initial: ')
            # print(non_zero_offs.loc[sample_ind])
            # find slack indices by finding the remainder of the current offer
            # divided by the current value then dividing this value by the value of the
            # current offer and taking indices where this is below some threshold
            # arbitrary .01 for now
            slack = (non_zero_offs % curr_val) / curr_val
            print('First slack: ')
            # print(slack.loc[sample_ind])
            # for slack greater than 50 %, subtract from 1 (since this implies the
            # original value is closer to the next factor of the current divisor than
            # the one immediately below it, so it may be in its rounding range, even
            # if not in that of the lower value)
            high_slack_inds = slack[slack > .5].index
            slack.loc[high_slack_inds] = 1 - slack.loc[high_slack_inds]
            print('Adjusted slack: ')
            # print(slack.loc[sample_ind])
            # truthiness check
            # print(slack.loc[sample_ind] == .01)
            # subset to slack less than .01 of rounding point
            print('adjusted')
            slack = slack[slack <= .011]
            print('Subset slack')
            # print(sample_ind[sample_ind.isin(slack.index)])
            # print(slack.loc[sample_ind[sample_ind.isin(slack.index)]])
            # separate the indices where slack is 0 and non-zero
            zero_slack = slack[slack == 0].index
            non_zero_slack = slack[slack > 0].index
            # activate the indicator for the non-zero and zero values
            curr_ser.loc[zero_slack] = 1
            curr_ser.loc[non_zero_slack] = 1

            # add the 0 slack indices to a running list of 0 slack indices and
            # the non-zero indices to a running list of non-zero indices for the current offer
            all_zero_inds = np.append(all_zero_inds, zero_slack.values)
            all_nonzero_inds = np.append(
                all_nonzero_inds, non_zero_slack.values)
            # finally, add the indicator for the current round-offer pair to the
            # data frame under the name created for it previously
            df[new_feat_name] = curr_ser

        # activate these indices in the slack series for the current offer
        slack_ser.loc[all_nonzero_inds] = 1
        # finally, add this series to the data frame with the name created for it
        # above
        df[slack_ser_name] = slack_ser
    # return the data frame with the new features
    return df


def get_prev_offr(turn):
    '''
    Description: get the column name of the previous turn made 
    by the player for whom we're predicting the next
    turn. If the current turn we're predicting is the seller's
    first turn, return 'start_price_usd'. If the current 
    turn we're predicting is the buyer's first turn,  
    '''
    turn_num = int(turn[1])
    turn_type = turn[0]
    if turn == 'start_price_usd':
        prev_turn = ''
    elif turn_type == 's':
        prev_turn = 'b' + str(turn_num)
    elif turn_type == 'b':
        if turn_num == 0:
            prev_turn = 'start_price_usd'
        else:
            prev_turn = 's' + str(turn_num - 1)
    if 'start_price_usd' not in prev_turn and prev_turn != '':
        prev_turn = 'offr_' + prev_turn
    return prev_turn


def get_ref_cols(df, turn):
    '''
    Description: Extracts the columns required to reproduce the actual 
    value of the of the response column in dollars as a data frame 
    and attaches these to the input data frame as ref_offr_rec and ref_offr_old
    Ref offer recent always refers to offr_turn and  ref offer old always 
    corresponds to the previous offer made by the party for whom we 
    are predicting the next offer
    Input: 
        df: pd.DataFrame containing bargaining features
        turn: string giving the name of the current turn
    Output: df containing all the original features and new reference columns
    '''
    ref_offr_rec = df['offr_%s' % turn]
    df['ref_rec'] = ref_offr_rec
    prev_offr = get_prev_offr(turn)
    resp_offr = get_resp_offr(turn)
    resp_offr = df[resp_offr]
    if prev_offr != '':
        ref_offr_old = df[prev_offr]
        df['ref_old'] = ref_offr_old
    df['ref_resp'] = resp_offr
    return df


def remove_oob(df, turn):
    # get a list of all offers
    offrs = get_all_offrs(turn)
    # remove starting price, so this list is reduced to all normalized offers
    offrs.remove('start_price_usd')
    # get total number of threads initially
    tot = len(df.index)
    # create running tally of number of threads dropped
    tally = 0
    # iterate over these offers
    for offr in offrs:
        # grab corresponding column
        offr_ser = df[offr]
        above = offr_ser[offr_ser > 1].index
        below = offr_ser[offr_ser < 0].index
        tally = tally + len(above) + len(below)
        df.drop(index=above, inplace=True)
        df.drop(index=below, inplace=True)
    print('OOB removed: %.2f %% ' % (tally/tot * 100))
    return df


def main():
    '''
    Description: imputes variables for nan values when reasonable to do so,
    removes all other nan columns, deletes all columns considered epistomological
    cheating (from the point of view of the buyer) for the time being,
    deletes all columns offers that do not have a reference price,
    deletes all date and time features
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
    df = pd.read_csv(read_loc, index_col=False)

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
        date_list.append('frac_remain_' + 'b' + str(i))
        date_list.append('frac_passed_' + 'b' + str(i))
        date_list.append('date_b' + str(i))
        # drop all seller dates
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

        # not leaving time features for observed turns
        if i > 0:
            date_list.append('time_' + 'b' + str(i))
        date_list.append('time_s' + str(i))

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
    df.drop(columns=['count2', 'count3', 'count4', 'ship_time_fastest', 'ship_time_slowest', 'count1',
                     'ref_price2', 'ref_price3', 'ref_price4', 'decline_price', 'accept_price',
                     'unique_thread_id'
                     ], inplace=True)

    # dropping all threads that do not have ref_price1
    df.drop(df[np.isnan(df['ref_price1'].values)].index, inplace=True)

    # inspecting columns
    print(df.columns)

    # generate indicator variables for whether eacah offer is round or
    # not
    df = round_inds(df, [1, 5, 10, 25], turn)

    # create reference columns (ref_rec guaranteed and ref_old usually)
    # so that we can re-create the response offer
    df = get_ref_cols(df, turn)
    df = norm_by_recent_offers(df, turn)

    # remove threads with out of bounds offers, namely threads where
    # a normalized offer is not in range [0, 1]
    # these may be considered abhorrent threads (in some cases, these
    # result from improper data entry, ie duplicated rows being mistaken as
    # responses from the other party)
    # in most cases, these are a result of "unfaithful" bargaining, abandoning
    # agreement ranging during convergence
    df = remove_oob(df, turn)
    # saving cleaned data frame, dropping unique_thread_id
    save_loc = 'data/exps/' + exp_name + \
        '/' + turn + '/' + filename
    print(save_loc)
    df.to_csv(save_loc, index=False)

    # normalize in this script and save the columns required for reference
    # in another data frame


if __name__ == '__main__':
    main()