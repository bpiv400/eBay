import numpy as np
import pandas as pd
import sys
import os
import argparse
import math
import matplotlib.pyplot as plt
import gc

# NOTE: ADDRESS THE MESSAGE PROBLEM
# for the time being we remove message


def date_feats(feat_df, col_abv):
    '''
    Description: For a given offer, creates features describing how
    much time has passed since the beginning of the auction
    and how much time remains before the end of the auction. For
    both, gives raw amount of time in hours as well as time as
    a fraction of total auction duration. Additionally gives
    auction duration

    Input: data frame containing 'auct_start_dt', 'auct_end_dt',
    and 'date_(col_abv)'

    Output: data frame with duration features added as
    dates
    '''
    col_abv = '_' + col_abv
    # grab offer time
    colname = 'date' + col_abv
    off_series = feat_df[colname].values
    # grab auction post time
    post_series = feat_df['auct_start_dt'].values
    # grab auction expiration time
    close_series = feat_df['auct_end_dt'].values
    close_series = close_series + np.timedelta64(24, 'h')

    # get total duration in hours
    dur = (close_series - post_series).astype(int)/1e9/math.pow(60, 2)

    rem = (close_series - off_series).astype(int)/1e9/math.pow(60, 2)
    passed = (off_series - post_series).astype(int)/1e9/math.pow(60, 2)

    # creating series for each new feature
    duration = pd.Series(dur, index=feat_df.index)
    remain = pd.Series(rem, index=feat_df.index)
    passed_time = pd.Series(passed, index=feat_df.index)
    frac_passed = pd.Series(passed/dur, index=feat_df.index)
    frac_remain = pd.Series(remain/dur, index=feat_df.index)

    feat_df['frac_remain' + col_abv] = frac_remain
    feat_df['frac_passed' + col_abv] = frac_passed
    feat_df['passed' + col_abv] = passed_time
    feat_df['remain' + col_abv] = remain
    feat_df['duration'] = duration

    return feat_df


def get_time_mins(df, end_code, init_code):
    '''
    Description: Creates a feature in df that gives the amount of time
    in minutes one player in the thread took to respond to offr_(init_code)
    with offr_(end_code)
    '''
    init_date = df['date_' + init_code]
    rsp_date = df['date_' + end_code]

    # find the difference, in minutes, between the response time and the offer time
    diff = (rsp_date.values - init_date.values).astype(int) / \
        1e9/math.pow(60, 1)
    diff = pd.Series(diff, index=rsp_date.index)

    # add init_offr series as a new column, both should have the same
    # index, since all threads should still be present
    df['time_' + end_code] = diff
    return df


def grab_turn(df, turn, seller):
    if turn == 0:
        if not seller:
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
            df = df.xs(0, level='turn_count', drop_level=False).copy()
            df.reset_index(level='turn_count', inplace=True, drop=True)
            # rename offr_price to match current turn
            df.rename(columns={'offr_price': 'offr_b0'}, inplace=True)
            # drop previous offer column since this corresponds to start price
            # in all cases
            df.drop(columns=['prev_offr_price'], inplace=True)
            # rename resp_offr to match the fact that its the sellers first offer
            df.rename(columns={'resp_offr': 'offr_s0'}, inplace=True)
            # rename response_time to reflect the fact that it gives the
            # date (and time) of the seller's first offer
            df.rename(
                columns={'response_time': 'date_s0'}, inplace=True)
            # rename the src_cre_date feature to reflect the fact that
            # its the creation date for the buyers first offer
            df.rename(
                columns={'src_cre_date': 'date_b0'}, inplace=True)
            df.rename(columns={
                'remain': 'remain_b0',
                'passed': 'passed_b0',
                'frac_passed': 'frac_passed_b0',
                'frac_remain': 'frac_remain_b0',
            }, inplace=True)
            df = get_time_mins(df, 's0', 'b0')
            df.drop(columns=['status_id', 'offr_type_id'], inplace=True)
            return df
        else:
            f2 = grab_turn(df.copy(), 1, False)
            # may want to drop more dates
            f2.drop(columns=['offr_s1', 'date_s1', 'time_s1',
                             'frac_passed_b1',
                             'frac_remain_b1',
                             'remain_b1',
                             'passed_b1'], inplace=True)
            print(len(f2.index))
            thrd_len = df.groupby('unique_thread_id').count()['turn_count']
            # extract thread ids associated with each length thread
            # that can be associated with the second turn

            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
            len_2_ids = thrd_len[thrd_len == 2].index
            len2 = df.loc[len_2_ids].copy(deep=True)

            if len(len2.index) > 0:
                # from length 2 subset, grab threads that contain a seller counteroffer
                seller_counters = len2[len2['offr_type_id']
                                       == 2].index.labels[0]
                seller_counters = len2.index.levels[0][seller_counters]
                df = df.loc[seller_counters].copy()
                gc.collect()
                if len(df.index) > 0:
                    off1df = grab_turn(df.reset_index(
                        drop=False).copy(), 0, False)
                    # drop all initial turns
                    df.drop(index=0, level='turn_count', inplace=True)
                    df.reset_index(level='turn_count', drop=True, inplace=True)
                    off1df['offr_b1'] = df['resp_offr']
                    off1df['date_b1'] = df['response_time']
                    off1df = date_feats(off1df, 's0')
                    off1df = get_time_mins(off1df, 'b1', 's0')
                    del df
                    f2 = pd.concat([f2, off1df], sort=False)
                    print(len(f2.index))
            return f2
    elif turn == 1:
        if not seller:
            # grab first offr df
            off1df = grab_turn(df.copy(), turn - 1, seller)
            off1df = off1df[['date_b0', 'date_s0',
                             'offr_s0', 'offr_b0',
                             'frac_remain_b0',
                             'frac_passed_b0',
                             'time_s0',
                             'remain_b0',
                             'passed_b0']].copy()
            # count turns in each thread
            thrd_len = df.groupby('unique_thread_id').count()['turn_count']
            # extract thread ids associated with each length thread
            # that can be associated with the second turn
            len_2_ids = thrd_len[thrd_len == 2].index
            len_34_ids = thrd_len[thrd_len.isin([3, 4])].index
            len_5_ids = thrd_len[thrd_len.isin([5])].index
            len_6_ids = thrd_len[thrd_len.isin([6])].index
            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len2 = df.loc[len_2_ids].copy(deep=True)
            df.drop(index=len2.index, inplace=True)
            gc.collect()
            len34 = df.loc[len_34_ids].copy(deep=True)
            df.drop(index=len34.index, inplace=True)
            gc.collect()
            len5 = df.loc[len_5_ids].copy(deep=True)
            df.drop(index=len5.index, inplace=True)
            gc.collect()
            len6 = df.loc[len_6_ids].copy(deep=True)

            del df
            del len_2_ids
            del len_34_ids
            del len_5_ids
            del len_6_ids

            if len(len2.index) > 0:
                # from length 2 subset, grab threads that contain a seller counteroffer
                seller_counters = len2[len2['offr_type_id']
                                       == 2].index.labels[0]
                seller_counters = len2.index.levels[0][seller_counters]
                gc.collect()

                # now remove all corresponding rows, remaining dataframe only contains threads that
                # correspond to those where the buyer makes two offers & the last row is the
                # buyers second offer
                len2.drop(index=seller_counters,
                          level='unique_thread_id', inplace=True)
                del seller_counters
                gc.collect()
            else:
                len2 = None
            # moving on to len34
            # pattern: the buyer's second turn occurs on turn_count = 2 except when a seller
            # counter offer occurs at turn_count = 2
            if len(len34.index) > 0:
                # remove the fourth offer in each 4 len thread
                if len(thrd_len[thrd_len == 4].index) > 0:
                    len34.drop(index=[3], level='turn_count', inplace=True)

                # extract second turn in each thread
                middle_offrs = len34.xs(
                    1, level='turn_count', drop_level=False).copy()
                # thread ids for threads where the middle offer is a seller counter offer
                middle_offr_ids = middle_offrs[middle_offrs['offr_type_id']
                                               == 2].index.labels[0]
                middle_offr_ids = middle_offrs.index.levels[0][middle_offr_ids]

                other_offr_ids = middle_offrs[middle_offrs['offr_type_id']
                                              != 2].index.labels[0]
                other_offr_ids = middle_offrs.index.levels[0][other_offr_ids]
                del middle_offrs
                # for these threads, drop the middle observation
                middle_offr_ids = [(middle_offr_id, 1)
                                   for middle_offr_id in middle_offr_ids]
                other_offr_ids = [(other_offr_id, 2)
                                  for other_offr_id in other_offr_ids]
                len34.drop(index=middle_offr_ids, inplace=True)
                len34.drop(index=other_offr_ids, inplace=True)
                del middle_offr_ids
                del other_offr_ids
            else:
                len34 = None
            # moving onto len 5 df
            # split into threads where the last offer is a seller counter offer or not
            if len(len5.index) > 0:
                last_seller_threads = len5.xs(4, level='turn_count', drop_level=True)[
                    'offr_type_id']
                # threads where teh last offer is a seller
                seller_threads = last_seller_threads[last_seller_threads == 2].index
                # threads where the last offer is a buyer
                buyer_threads = last_seller_threads[last_seller_threads != 2].index
                del last_seller_threads
                # grab the corresponding feature sets from the df for buyers and sellers using
                # the ids above
                # in threads where the last offer is a buyer, there must have been exactly 2
                # seller offers, meaning turn_count=2 corresponds to the buyer's second turn
                buyers = len5.loc[buyer_threads].copy()
                sellers = len5.loc[seller_threads].copy()

                del len5

                # for the buyers df, throw out everything
                # moving on to 6 length
                if len(buyers.index) > 0:
                    buyers.drop(index=[1, 3, 4],
                                level='turn_count', inplace=True)

                # for the sellers df, if the first turn is declined, then the b1 is located
                # at turn_count = 1
                # find the subset of threads where turn_count = 0 is declined
                first_turn = sellers.xs(0, level='turn_count', drop_level=True)[
                    'status_id']
                first_turn_dec = first_turn[first_turn.isin(
                    [0, 2, 6, 8])].index
                # remove unnecessary variable
                del first_turn
                # extract corresponding feature sets from dual indexed df
                first_dec_threads = sellers.loc[first_turn_dec].copy()
                # drop all indices except turn_count = 0 (the initial offer) and turn_count = 1
                # which corresponds to b1
                first_dec_threads.drop(
                    index=[2, 3, 4], level='turn_count', inplace=True)
                # remove the isolated threads from the buyers df from which they were removed
                sellers.drop(index=first_turn_dec,
                             level='unique_thread_id', inplace=True)
                # this leaves only threads where the last offer is a seller and the
                # first offer is not declined
                # in these cases, remove all turns except turn_count = 0 and turn_count = 2, since turn_count=1
                # must be a counter offer
                sellers.drop(index=[1, 3, 4], level='turn_count', inplace=True)
            else:
                buyers = None
                sellers = None
                first_dec_threads = None
            # throw out everything past turn_count = 2 and the second turn
            # which necessarily must be a seller offer
            if len(len6.index) > 0:
                len6.drop(index=[1, 3, 4, 5],
                          level='turn_count', inplace=True)
            else:
                len6 = None
            # concat all dfs
            out = pd.concat([len2, len34, buyers, sellers,
                             first_dec_threads, len6], sort=False)
            del len2
            del len34
            del sellers
            del first_dec_threads
            del buyers
            del len6

            # grab all thread ids
            out.index = out.index.remove_unused_levels()
            all_threads = out.index.levels[0]
            # create list of tuples correponding to the first offer in each thread
            first_turn_ids = [(thread_id, 0) for thread_id in all_threads]
            out.drop(index=first_turn_ids, inplace=True)

            out.reset_index(level='turn_count', inplace=True, drop=True)
            if (len(np.unique(out.index.values)) != len(out.index)):
                raise ValueError('thread indices are not unique, uh oh')

            out = out.merge(off1df, how='inner',
                            left_index=True, right_index=True)

            # drop prev_offr_price since we already have it via offr_s0 in off1df
            out.drop(columns=['prev_offr_price'], inplace=True)

            # rename response_time -> 'date_s1'
            # rename src_cre_date -> 'date_b1'
            # rename offr_price -> 'offr_b1'
            # rename resp_offr -> 'offr_s1'
            out.rename(columns={'response_time': 'date_s1',
                                'src_cre_date': 'date_b1',
                                'offr_price': 'offr_b1',
                                'resp_offr': 'offr_s1'},
                       inplace=True)
            # grab the initial offer time
            out = date_feats(out, 's0')
            out = date_feats(out, 'b1')
            out = get_time_mins(out, 'b1', 's0')
            out = get_time_mins(out, 's1', 'b1')
            out.drop(columns=['frac_remain', 'frac_passed', 'passed', 'remain', 'status_id',
                              'offr_type_id'], inplace=True)
            return out
        else:
            f2 = grab_turn(df.copy(), 2, False)
            f2.drop(columns={
                'date_s2', 'offr_s2', 'time_s2',
                'frac_remain_b2',
                'frac_passed_b2',
                'passed_b2', 'remain_b2',
            }, inplace=True)

            # count turns in each thread
            thrd_len = df.groupby('unique_thread_id').count()['turn_count']
            # extract thread ids associated with each length thread
            # that can be associated with the second turn
            len_3_ids = thrd_len[thrd_len == 3].index
            len_4_ids = thrd_len[thrd_len == 4].index
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            len3 = df.loc[len_3_ids].copy(deep=True)
            len4 = df.loc[len_4_ids].copy(deep=True)
            del df

            if len(len3.index) > 0:
                last_turn_3 = len3.xs(
                    2, level='turn_count', drop_level=True).copy()
                last_turn_3 = last_turn_3[last_turn_3['offr_type_id'] == 2].index
                len3 = len3.loc[last_turn_3].copy()
                len3_comp = grab_turn(len3.reset_index(
                    drop=False).copy(), 1, False)
                len3 = len3.xs(2, level='turn_count', drop_level=True).copy()
                len3_comp['offr_b2'] = len3['resp_offr']
                len3_comp['date_b2'] = len3['response_time']
                len3_comp = date_feats(len3_comp, 's1')
                len3_comp = get_time_mins(len3_comp, 'b2', 's1')
                del len3
                del last_turn_3
            else:
                len3_comp = None

            if len(len4.index) > 0:
                sec_turn = len4.xs(1, level='turn_count',
                                   drop_level=True).copy()
                last_turn = len4.xs(3, level='turn_count',
                                    drop_level=True).copy()
                sec_turn_ids = sec_turn['offr_type_id'] == 2
                last_turn_ids = last_turn['offr_type_id'] == 2
                shared = sec_turn_ids & last_turn_ids
                sec_turn_ids = sec_turn_ids[shared].index
                len4 = len4.loc[sec_turn_ids].copy()
                len4_comp = grab_turn(len4.reset_index(
                    drop=False).copy(), 1, False)
                len4 = last_turn
                len4_comp['offr_b2'] = len4['resp_offr']
                len4_comp['date_b2'] = len4['response_time']
                len4_comp = date_feats(len4_comp, 's1')
                len4_comp = get_time_mins(len4_comp, 'b2', 's1')
            else:
                len4_comp = None

            out = pd.concat([f2, len4_comp, len3_comp], sort=False)
            return out
    else:
        if not seller:
            # extract second offer df before doing anything else
            off2df = grab_turn(df.copy(), 1, seller)
            off2df = off2df[['offr_b0', 'offr_b1',
                             'date_b0', 'date_b1', 'date_s1', 'date_s0',
                             'offr_s0', 'offr_s1', 'frac_passed_b0',
                             'frac_passed_b1', 'frac_passed_s0',
                             'frac_remain_b0', 'frac_remain_b1',
                             'frac_remain_s0', 'time_s0', 'time_s1',
                             'time_b1',
                             'passed_b0', 'passed_b1', 'passed_s0',
                             'remain_b0', 'remain_b1',
                             'remain_s0']].copy()
            # print(off2df)

            thrd_len = df.groupby('unique_thread_id').count()['turn_count']

            len_3_ids = thrd_len[thrd_len == 3].index
            len_4_ids = thrd_len[thrd_len == 4].index
            len_5_ids = thrd_len[thrd_len == 5].index
            len_6_ids = thrd_len[thrd_len == 6].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len3 = df.loc[len_3_ids].copy()
            len4 = df.loc[len_4_ids].copy()
            len5 = df.loc[len_5_ids].copy(deep=True)
            df.drop(index=len5.index, inplace=True)
            gc.collect()
            len6 = df.loc[len_6_ids].copy(deep=True)

            # clean up crew
            del df
            del len_3_ids
            del len_4_ids
            del len_5_ids
            del len_6_ids

            # focusing on len3--we only want threads where the
            # second and third offers are both buyer offers
            # extracting threads where the second offer is a seller offer
            if len(len3.index) > 0:
                sec_off = len3.xs(1, level='turn_count',
                                  drop_level=True)['offr_type_id']
                seller_threads = sec_off[sec_off == 2].index

                # remove these threads from the df
                len3.drop(index=seller_threads,
                          level='unique_thread_id', inplace=True)
                del sec_off
                del seller_threads

                # extracting threads where the third offer is a seller offer
                third_off = len3.xs(2, level='turn_count', drop_level=True)[
                    'offr_type_id']
                seller_threads = third_off[third_off == 2].index
                # remove these threads from the df
                len3.drop(index=seller_threads,
                          level='unique_thread_id', inplace=True)
                del seller_threads
                # from the remaining threads, remove the first and second offers
                len3.drop(index=[0, 1], level='turn_count', inplace=True)
            else:
                len3 = None
            if len(len4.index) > 0:
                # moving on to length 4 threads
                # extracting fourth offer from each thread
                last = len4.xs(3, level='turn_count', drop_level=True)[
                    'offr_type_id']

                # threads where the last offer is made by a seller
                last_seller = last[last == 2].index
                # print(last_seller[0:10])
                del last

                # threads where the second offer is made by a seller
                two = len4.xs(1, level='turn_count', drop_level=True)[
                    'offr_type_id']
                two_seller = two[two == 2].index
                # print(two_seller[0:10])

                # from the remaining threads, remove the first and second offers
                len4.drop(index=[0, 1], level='turn_count', inplace=True)
                del two

                # get threads where both second and last offers are made by the seller
                shared = np.intersect1d(last_seller.values, two_seller.values)
                del two_seller

                # drop all threads where both the second and the last offer
                # are made by the seller
                len4.drop(index=shared, level='unique_thread_id', inplace=True)
                del shared

                # get all threads where the last offer is made by the seller but
                # the second offer is not (ie remaining threads where second offer
                # is made by the seller)
                # these are the threads where the last buyer offer is located
                # at turn 3 (ie turn_count = 2)

                remaining = len4.index.labels[0]
                remaining = len4.index.levels[0][remaining]
                # print(remaining[0:10])

                last_seller = np.intersect1d(remaining, last_seller.values)

                # get all other threads
                remaining = np.setdiff1d(remaining, last_seller)
                remaining = [(thread_id, 3) for thread_id in remaining]
                last_seller = [(thread_id, 2) for thread_id in last_seller]
                all_ids = remaining + last_seller
                # print(len(all_ids))
                del last_seller
                del remaining

                len4 = len4.loc[all_ids].copy()
                # print(len4)
                # print(len(len4.index))
            else:
                len4 = None
            if len(len5.index) > 0:
                # moving onto len 5 df
                # split into threads where the last offer is a seller counter offer or not
                last_seller_threads = len5.xs(4, level='turn_count', drop_level=True)[
                    'offr_type_id']
                # threads where teh last offer is a seller
                seller_threads = last_seller_threads[last_seller_threads == 2].index
                # threads where the last offer is a buyer
                buyer_threads = last_seller_threads[last_seller_threads != 2].index
                del last_seller_threads
                # grab the corresponding feature sets from the df for buyers and sellers using
                # the ids above
                # in threads where the last offer is a buyer, there must have been exactly 2
                # seller offers, meaning turn_count=4 corresponds to the buyer's third turn
                buyers = len5.loc[buyer_threads].copy()
                sellers = len5.loc[seller_threads].copy()
                del len5

                # for the buyers df, throw out everything except turn 4
                if len(buyers.index) > 0:
                    buyers.drop(index=[0, 1, 2, 3],
                                level='turn_count', inplace=True)
                else:
                    buyers = None
                # for the seller's df, throw out everything except turn 3
                if len(sellers.index) > 0:
                    sellers.drop(index=[0, 1, 2, 4],
                                 level='turn_count', inplace=True)
                else:
                    sellers = None
            else:
                buyers = None
                sellers = None

            if len(len6.index) > 0:
                len6.drop(index=[0, 1, 2, 3, 5],
                          level='turn_count', inplace=True)
            else:
                len6 = None
            out = pd.concat([len6, len4, len3, buyers, sellers])
            # cleaning unused vars
            del len4
            del len3
            del len6
            del buyers
            del sellers

            # rename columns
            out.rename(columns={'response_time': 'date_s2',
                                'src_cre_date': 'date_b2',
                                'offr_price': 'offr_b2',
                                'resp_offr': 'offr_s2'},
                       inplace=True)
            # drop extra columns
            out.drop(columns=['prev_offr_price', 'status_id',
                              'offr_type_id'], inplace=True)
            # adding features
            out.reset_index(level='turn_count', drop=True, inplace=True)

            out = out.merge(off2df, how='inner',
                            left_index=True, right_index=True)
            del off2df
            out = date_feats(out, 's1')
            out = date_feats(out, 'b2')
            out = get_time_mins(out, 'b2', 's1')
            out = get_time_mins(out, 's2', 'b2')
            out.drop(columns=['passed', 'frac_passed',
                              'remain', 'frac_remain'], inplace=True)
            return out

        else:
            thrd_len = df.groupby('unique_thread_id').count()['turn_count']

            # for the s2 model, we are predicting 'offr_b3'--since buyers
            # don't technically have the ability to make a 4th offer,
            # this offer can only be a response in the threads where the seller
            # makes the last offer

            # first, grab all threads where the buyer makes at least 3 offers
            b2df = grab_turn(df.copy(), 2, seller=False)
            # retain only columns derived from offer-by-offer processing
            b2df = b2df[['offr_b0', 'offr_b1', 'offr_b2', 'offr_s2',
                         'date_b0', 'date_b1', 'date_b2',
                         'date_s1', 'date_s0', 'date_s2',
                         'offr_s0', 'offr_s1', 'frac_passed_b0',
                         'frac_passed_b1', 'frac_passed_b2',
                         'frac_passed_s0', 'frac_passed_s1',
                         'frac_remain_b0', 'frac_remain_b1',
                         'frac_remain_s0', 'frac_remain_b2',
                         'frac_remain_s1', 'time_s0', 'time_s1',
                         'time_b1', 'time_b2', 'time_s2',
                         'passed_b0', 'passed_b1', 'passed_s0',
                         'passed_b2', 'passed_s1',
                         'remain_b0', 'remain_b1',
                         'remain_b2', 'remain_s1',
                         'remain_s0']].copy()
            # error checking

            # create subset dfs that only contain threads with 4, 5, or 6 offers (the minimum required for the
            # seller to make a counter offer)
            len_4_ids = thrd_len[thrd_len == 4].index
            len_5_ids = thrd_len[thrd_len == 5].index
            len_6_ids = thrd_len[thrd_len == 6].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len4 = df.loc[len_4_ids].copy(deep=True)
            df.drop(index=len4.index, inplace=True)
            gc.collect()
            len5 = df.loc[len_5_ids].copy(deep=True)
            df.drop(index=len5.index, inplace=True)
            gc.collect()
            len6 = df.loc[len_6_ids].copy(deep=True)

            # clean up crew
            del df
            del len_4_ids
            del len_5_ids
            del len_6_ids

            # now, subset len4 to contain only threads where the last offer is a seller offer
            # and the first offer is declined
            if len(len4.index) > 0:
                first_off_stat = len4.xs(0, level='turn_count',
                                         drop_level=True)['status_id'].copy()
                last_off_type = len4.xs(3, level='turn_count',
                                        drop_level=True)['offr_type_id'].copy()
                # get indices where the first offer is declined
                f_dec_inds = first_off_stat[first_off_stat.isin(
                    [0, 2, 6, 8])].index
                # get indices where the last offer is a seller offer
                l_sel_inds = last_off_type[last_off_type == 2].index
                # clean up crew
                del first_off_stat
                del last_off_type
                # create a set of thread ids where first_offer is declined and
                # the last offer is a seller offer
                shared_inds = np.intersect1d(
                    f_dec_inds.values, l_sel_inds.values)
                del f_dec_inds
                del l_sel_inds

                # now subset len4 df to only contain these threads
                len4 = len4.loc[shared_inds].copy()
                # remove all but the last offer for each data frame
                len4.drop(index=[0, 1, 2], level='turn_count', inplace=True)
            else:
                len4 = None

            # subset len5 df to only contain threads where the last offer is a seller
            if len(len5.index) > 0:
                last_off_type = len5.xs(4, level='turn_count',
                                        drop_level=True)['offr_type_id'].copy()
                l_sel_inds = last_off_type[last_off_type == 2].index
                del last_off_type
                len5 = len5.loc[l_sel_inds]
                del l_sel_inds
                len5.drop(index=[0, 1, 2, 3],
                          level='turn_count', inplace=True)
            else:
                # remove all but the last offer for each data frame
                len5 = None
            if len(len6.index) > 0:
                len6.drop(index=[0, 1, 2, 3, 4],
                          level='turn_count', inplace=True)
            else:
                len6 = None

            out = pd.concat([len4, len5, len6])

            out.rename(columns={'response_time': 'date_b3',
                                'resp_offr': 'offr_b3'},
                       inplace=True)
            out.drop(columns=['prev_offr_price', 'status_id', 'passed', 'frac_passed',
                              'remain', 'frac_remain',
                              'offr_type_id', 'src_cre_date', 'offr_price'], inplace=True)
            # adding features
            out.reset_index(level='turn_count', drop=True, inplace=True)
            out = out.merge(b2df, how='inner',
                            left_index=True, right_index=True)
            out = date_feats(out, 's2')
            out = get_time_mins(out, 'b3', 's2')
            return out


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


def check_legit(df, last_offr_code, feats=['frac_passed', 'frac_remain', 'passed', 'remain', 'offr', 'date', 'time']):
    '''
    Description: ensure the legitimacy of the return data frame, meaning
        1. no NaN or NA values and
        2. no repeated threads in the index
        3. that the data frame is indexed by unique therad id
        4. The data frame contains all the expected features
        5. That the data frame does not contain unique_thread_id or turn_count as features
        6. That the data frame does not contain offr_type_id or status_id as features

    Otherwise, throws a value Error
    Input:
        df: pandas.DataFrame to be checked
        last_offr_code: string giving the offer code of the last offer the
        data frame is expected to containing
        feats: list of strings giving the features whose existence is expected
    Returns: None
    '''
    # grab codes for all offers up until the current turn
    offr_codes = all_offr_codes(last_offr_code)
    print('Checking data frame for sequences ending with %s' % last_offr_code)
    # grab columns
    cols = df.columns
    # create a boolean to track whether an engineered column contains null values
    null_found = False
    # iterate over all codes
    for code in offr_codes:
        # iterate over all features whose existence we must ensure
        for feat in feats:
            # ensure the column is in the data frame, raise an error otherwise
            # and if each column is contained, ensure that it does not have
            # any null values
            feat_name = '%s_%s' % (feat, code)
            if feat_name not in cols:
                raise ValueError(
                    'df expected %s, but does not contain it' % feat_name)
            # if the column exists
            else:
                # check if the current column has any null values
                has_null = df[feat_name].isnull().any()
                # if a null value is detected, throw a value exception
                if has_null:
                    print('Null vals contained in %s' % feat_name)
                    null_found = True
    # after looping over all columns, if a null value has been discovered,
    # stop execution
    if null_found:
        raise ValueError('Df contains null value in an engineering column')
    # check that df is indexed by unique_thread_id and turn_count doesn't exist in
    # the data frame
    if df.index.name != 'unique_thread_id':
        raise ValueError(
            'Expected df to be indexed by unique_thread_id, but was indexed by %s' % df.index.name)
    # check that df does not contain a turn_count column
    if 'turn_count' in cols or 'unique_thread_id' in cols:
        raise ValueError(
            'turn_count and unique_thread_id should not be columns')
    # check that each thread id in the index is unique
    if len(df.index) != len(np.unique(df.index.values)):
        raise ValueError('index contains repeating thread ids')
    # check that the df does not contain any nan values

    if 'offr_type_id' in cols or 'status_id' in cols:
        raise ValueError('DF contains status id or offr_type_id')

    # ensure no collision columns exist in the output df
    for col in cols:
        if '_x' in col or '_y' in cols:
            print('%s collided in output df' % col)
            raise ValueError('DF contains column collision')
    pass


def grab_seqs_len2(df):
    '''
    Description: grabs the sequences where only two rounds of bargaining
    take place, i.e. the buyer makes an offer then the seller makes an offer

    This corresponds to only threads where the buyer makes an offer
    then the seller accepts or rejects them, then the thread ends

    All threads of length 1 in the data_frame characterize this
    '''
    # grab the length of all threads in the original data frame
    # by counting the rows with each thread id
    thrd_len = df.groupby('unique_thread_id').count()['turn_count']
    # isolate the threads of length 1 by dropping all other threads from the
    # data frame
    # first grab the threads
    seq_threads = thrd_len[thrd_len != 1].index
    # set index to tuple of thread_id and turn_count, since this pair should
    # constitute a unique identifier
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    # drop the entries with the extracted thread ids
    df.drop(index=seq_threads, level='unique_thread_id', inplace=True)
    # drop the turn_count index
    df.reset_index(level='turn_count', drop=True, inplace=True)
    # rename offer features to identify the corresponding
    df.rename(columns={'response_time': 'date_s0',
                       'offr_price': 'offr_b0',
                       'resp_offr': 'offr_s0',
                       'src_cre_date': 'date_b0'}, inplace=True)
    # grab date features for the first two offers
    df = date_feats(df, 'b0')
    df = date_feats(df, 's0')
    # get time feature for the first seller offer
    df = get_time_mins(df, 's0', 'b0')
    # set time feature for the first buyer offer to equal the amount
    # of time that has passed since the beginning of the auction
    df['time_b0'] = df['passed_b0']

    # drop status id and offr_type features
    df.drop(columns=['offr_type_id', 'status_id'], inplace=True)
    # drop the prev_offr_price feature...meaningless for b0
    df.drop(columns='prev_offr_price', inplace=True)
    # execute error checking function
    # remove lagging columns
    if 'passed' in df.columns:
        df.drop(columns=['passed', 'frac_passed',
                         'remain', 'frac_remain'], inplace=True)
    check_legit(df, 's0')
    # print(df.columns)
    return df


def grab_seqs_len3(df):
    '''
    Description: grabs the sequences where only three rounds of bargaining
    take place...
    This corresponds to threads where the buyer makes an offer, the seller
    makes an offer, then the buyer accepts or rejects this offer, ending the
    thread
    '''
    # extract ids of threads where only two offers are made and the second
    # offer is made by the seller
    thrd_len = df.groupby('unique_thread_id').count()['turn_count']
    seq_threads = thrd_len[thrd_len != 2].index

    # set index to tuple of thread_id and turn_count, since this pair should
    # constitute a unique identifier
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    # drop all seq threads, leaving only threads of length 2
    df.drop(index=seq_threads, level='unique_thread_id', inplace=True)
    # extract the offr type id for the second turn in each remaining thread
    type_ids = df.xs(1, level='turn_count',
                     drop_level=True)['offr_type_id'].copy()
    # get the thread ids of the threads where the second offer isn't a seller offer
    not_seller_threads = type_ids[type_ids != 2].index
    # drop these threads from the data frame
    df.drop(index=not_seller_threads, level='unique_thread_id', inplace=True)

    # grab first turn information corresponding to threads in the remaining
    # subset (only those of len 3 exactly) -- should reset index completely
    # before feeding to grab_turn
    off1df = grab_turn(df.reset_index(
        drop=False, inplace=False), 0, False)
    # drop all initial turns
    df.drop(index=0, level='turn_count', inplace=True)
    # drop turn_count as an index
    df.reset_index(level='turn_count', drop=True, inplace=True)
    # set the response offer as the buyers second
    off1df['offr_b1'] = df['resp_offr']
    # set the date of the response offer to be the date of the buyers second
    off1df['date_b1'] = df['response_time']
    #############################################
    # NOTE NOTE NOTE NOTE NOTE NOTE NOTE
    # an indicator for defection can be set here
    # before dropping status id and offer type
    # if we want to implement that later
    # Such an indicator would be activated on all
    # threads of length 3 exactly where the
    # buyer does not accept the seller's last
    # offer -- use status id of df since this frame
    # only contains the sellers last offer for each
    # thread
    ###########################################
    # remove df
    del df
    # add date features for s0 (left out in grab_turn(...), since
    # the function was made not to generate date features for
    # response offers
    off1df = date_feats(off1df, 's0')
    off1df = date_feats(off1df, 'b1')
    off1df = get_time_mins(off1df, 'b1', 's0')
    off1df['time_b0'] = off1df['passed_b0']

    if 'status_id' in off1df.columns:
        off1df.drop(columns=['status_id'], inplace=True)
    if 'offr_type_id' in off1df.columns:
        off1df.drop(columns=['offr_type_id'], inplace=True)

    # remove lagging columns
    if 'passed' in off1df.columns:
        off1df.drop(columns=['passed', 'frac_passed',
                             'remain', 'frac_remain'], inplace=True)

    check_legit(off1df, 'b1')
    return off1df


def grab_seqs_len4(df):
    '''
    Description: grabs the sequences where only four rounds of bargaining
    take place...
    This corresponds to all threads returned by grab_turn(df, 1, False)
    except those where more than 4 offers have taken place. To isolate these,
    we remove threads where:
        -  four or more offers have been recorded in the original data frame
        We remove all threads where four offers have been recorded, since these at
        least include at least four distinct offers and implicitly, a response to the
        last offer, totaling 5 offers...the same argument extends to threads
        with more offers obviously
        - threads where we have 3 recorded offers and the buyer is responsible for
        all of them
        - threads where we have 3 recorded offers and the seller is responsible for the
        last one. This can only occur when the seller rejects the buyers first offer
        and counters the second one. As a result, the response to the last
        offer corresponds to the fifth offer
    '''
    # grab thrd length for each thread
    thrd_len = df.groupby('unique_thread_id').count()['turn_count']
    # get long threads (for removal)
    long_thrds = thrd_len[thrd_len >= 4].index
    # get threads of length 3
    len3_thrds = thrd_len[thrd_len == 3].index
    # set index to tuple of turn count and thread id
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    # grab the offr type of the last turn for the threads with 3 offers
    final_turn_type = df.loc[len3_thrds].xs(2, level='turn_count', drop_level=True)[
        'offr_type_id'].copy()
    # some error checking for the subset above
    if len(df.index) == len(final_turn_type.index):
        raise ValueError('Subset failure')
    # get ids of threads where the last offer is a seller offer
    final_turn_seller = final_turn_type[final_turn_type == 2].index
    # get the ids of the len3 threads where the last offer is a buyer
    final_turn_buyer = final_turn_type[final_turn_type != 2].index
    # grab the middle turn for the these threads
    mid_turn_type = df.loc[final_turn_buyer].xs(
        1, level='turn_count', drop_level=True)['offr_type_id'].copy()
    # get indices where the middle turn of the length 3 thread is also a buyer
    all_buyers = mid_turn_type[mid_turn_type != 2].index
    # drop len3 threads where the last offer is made by a seller
    df.drop(index=final_turn_seller, level='unique_thread_id', inplace=True)
    # drop len3 threads where all offers are made by the buyer
    df.drop(index=all_buyers, level='unique_thread_id', inplace=True)
    # drop all threads longer than 3 offers
    df.drop(index=long_thrds, level='unique_thread_id', inplace=True)
    # grab thread offer features after resetting index
    df.reset_index(drop=False, inplace=True)
    df = grab_turn(df, 1, False)
    # add features for s1 and a filler time for b0
    df = date_feats(df, 's1')
    df['time_b0'] = df['passed_b0']

    # remove lagging columns
    if 'passed' in df.columns:
        df.drop(columns=['passed', 'frac_passed',
                         'remain', 'frac_remain'], inplace=True)

    # finally check the data for processing errors
    check_legit(df, 's1')
    return df


def grab_seqs_len5(df):
    '''
    Description: grabs the sequences where only five rounds of bargaining
    take place, ie threads where the last turn is b2
    This corresponds to all threads where that would be returned by
    grab_turn(df, 1, True) after removing all threads where 6 or 7 offers
    take place...This corresponds to removing:
    -all threads of with 6 offers stored
    -all threads of length 3 where all offers are made by buyers
    -all threads of length 4 where at least 3 offers are buyer offers
    -all threads where at least 5 offers are made
    '''
    thrd_len = df.groupby('unique_thread_id').count()['turn_count']
    long_thrds = thrd_len[thrd_len >= 5].index
    len3_thrds = thrd_len[thrd_len == 3].index
    len4_thrds = thrd_len[thrd_len == 4].index
    # set index to turn count, thread_id
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    # grab just offer_type_id and unique_thread_id columns for threads of length 4
    len4_type = df.loc[len4_thrds, ['offr_type_id']].copy()
    # subset to contain only buyer offers
    len4_type = len4_type[len4_type['offr_type_id'] != 2]
    # drop unique_thread_id and turn_count as an index
    len4_type.reset_index(drop=False, inplace=True)
    # count turns for each unique thread id--this amounts to counting buyer turns
    # since seller turns have been removed
    byr_cnt = len4_type.groupby('unique_thread_id').count()['turn_count']
    # get all threads where 3 offers ha e been made by buyers
    three_byr_thrds = byr_cnt[byr_cnt == 3].index
    # print(three_byr_thrds)

    # get offer type for threads of length 3
    len3_type = df.loc[len3_thrds, ['offr_type_id']].copy()
    # subset to contain only buyer offers
    len3_type = len3_type[len3_type['offr_type_id'] != 2]
    # drop unique_thread_id and turn_count as an index
    len3_type.reset_index(drop=False, inplace=True)
    # count turns for each unique thread id--this amounts to counting buyer turns
    # since seller turns have been removed
    byr_cnt = len3_type.groupby('unique_thread_id').count()['turn_count']
    # get threads where all offers are made by buyer
    all_byr_thrds = byr_cnt[byr_cnt == 3].index

    # drop threads of length 4 where 3 offers have been made by buyers
    df.drop(index=three_byr_thrds, level='unique_thread_id', inplace=True)
    # drop threads of length 3 where all offers have been made by buyers
    df.drop(index=all_byr_thrds, level='unique_thread_id', inplace=True)
    # drop threads of length 5 or 6, since these implicitly must have at least 6 offers
    df.drop(index=long_thrds, level='unique_thread_id', inplace=True)
    # reset index
    df.reset_index(drop=False, inplace=True)
    # input remaining data to grab_turn for seller=T, turn = 1
    df = grab_turn(df, 1, True)
    # add date features for b2
    df = date_feats(df, 'b2')
    df['time_b0'] = df['passed_b0']

    # remove lagging columns
    if 'passed' in df.columns:
        df.drop(columns=['passed', 'frac_passed',
                         'remain', 'frac_remain'], inplace=True)

    # check validity
    check_legit(df, 'b2')

    return df


def grab_seqs_len6(df):
    '''
    Description: grabs the sequences where only six rounds of bargaining
    take place, ie threads where the last turn is b2
    This corresponds to all threads where that would be returned by
    grab_turn(df, 2, False) after removing all threads where 7 offers
    take place, namely:
    -threads of length 4 where the first 3 offers are made by the buyer, and
    the last offer is made by the seller
    -threads of length 5 where the last offer is made by the seller
    - all threads of length 6
    '''
    # group by thread length
    thrd_len = df.groupby('unique_thread_id').count()['turn_count']
    # get thread ids for len 4, 5, and 6
    len6_thrds = thrd_len[thrd_len == 6].index
    len5_thrds = thrd_len[thrd_len == 5].index
    len4_thrds = thrd_len[thrd_len == 4].index
    # set index to unique_thread_id, turn_count
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    # find the set of len 4 threads where the first three offers are made
    # by buyers and the last offer is made by a seller
    # grab just offer_type_id and unique_thread_id columns for threads of length 4
    len4_type = df.loc[len4_thrds, ['offr_type_id']].copy()
    len4_last_type = len4_type.xs(
        3, level='turn_count', drop_level=True).copy()
    # threads where the type of the last offer is a seller offer
    seller_last_thrds = len4_last_type[len4_last_type['offr_type_id'] == 2].index
    # now remove all seller offers from the len4_type data frame
    len4_type = len4_type[len4_type['offr_type_id'] != 2]
    # reset index so turn_count and unique_thread_id both become columns
    len4_type.reset_index(drop=False, inplace=True)
    byr_cnt = len4_type.groupby('unique_thread_id').count()['turn_count']
    # get thread_ids for all threads with 3 byr offers
    three_byr_thrds = byr_cnt[byr_cnt == 3].index
    # find the len4 threads where there are 3 buyer offers and the last offer is made by the seller
    three_byr_last_seller_thrds = np.intersect1d(
        three_byr_thrds.values, seller_last_thrds.values)

    # collect unused variables
    del byr_cnt
    del len4_type
    del len4_last_type
    del seller_last_thrds
    del three_byr_thrds
    del len4_thrds

    # find the type of the last offer for threads of len 5
    len5_type = df.loc[len5_thrds, ['offr_type_id']].copy()
    len5_last_type = len5_type.xs(
        4, level='turn_count', drop_level=True).copy()
    # threads where the type of the last offer is a seller offer
    seller_last_thrds = len5_last_type[len5_last_type['offr_type_id'] == 2].index

    # drop the len 4 threads with 3 buyers and 1 seller as the last offer from the data frame
    df.drop(index=three_byr_last_seller_thrds,
            level='unique_thread_id', inplace=True)
    # drop the len 5 threads that end in a seller turn from the data frame
    df.drop(index=seller_last_thrds, level='unique_thread_id', inplace=True)
    # drop len 6 threads from the data frame
    df.drop(index=len6_thrds, level='unique_thread_id', inplace=True)

    # reset index to prepare for grab_turn
    df.reset_index(drop=False, inplace=True)

    # execute turn grab where the last observed offer is the sellers second offer
    df = grab_turn(df, 2, False)
    # add date features for response variable
    df = date_feats(df, 's2')
    df['time_b0'] = df['passed_b0']

    # remove lagging columns
    if 'passed' in df.columns:
        df.drop(columns=['passed', 'frac_passed',
                         'remain', 'frac_remain'], inplace=True)

    # check validity
    check_legit(df, 's2')
    return df


def grab_seqs_len7(df):
    '''
    Description: grabs the sequences where 7 rounds of bargaining
    take place, ie those where the last offer is b3
    This is equivalent to all the threads returned by grab_turn(df, 2, True)
    '''
    # grab thread and offer features
    df = grab_turn(df, 2, True)
    # create new date features
    df = date_feats(df, 'b3')
    df['time_b0'] = df['passed_b0']

    # remove lagging columns
    if 'passed' in df.columns:
        df.drop(columns=['passed', 'frac_passed',
                         'remain', 'frac_remain'], inplace=True)

    # check for legitimacy
    check_legit(df, 'b3')
    return df


def grab_seqs(df):
    '''
    Description: Converts a data frame of offers to a data frame where
    each row with id (len, thread_id) describes a bargaining sequence
    of length (len). Columns contain all offers necessary for the
    longest sequence (through b3)
    '''
    # df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    seqs2 = grab_seqs_len2(df.copy())
    seqs3 = grab_seqs_len3(df.copy())
    seqs4 = grab_seqs_len4(df.copy())
    seqs5 = grab_seqs_len5(df.copy())  # may contain the same index as
    seqs6 = grab_seqs_len6(df.copy())  # seq 6 -- take a closer look
    seqs7 = grab_seqs_len7(df.copy())

    del df
    # some error checking
    # concatenate ids from all sequence dfs
    # ensure there are no repeats
    all_ids = seqs2.index.values
    ######################################################
    # Error checking
    # print(type(all_ids))
    # print(type(all_ids[0]))
    # print(type(seqs3.index.values))
    # print(type(seqs3.index.values[0]))
    ######################################################

    all_ids = np.concatenate((all_ids, seqs3.index.values))

    # compare indices of len2, len3, len4, and len5 to len6
    #####################################################################
    # ERROR CHECKING
    if np.intersect1d(seqs2.index.values, seqs6.index.values).size > 0:
        raise ValueError('len2 and len6 ids collide')
    if np.intersect1d(seqs3.index.values, seqs6.index.values).size > 0:
        raise ValueError('len3 and len6 ids collide')
    if np.intersect1d(seqs4.index.values, seqs6.index.values).size > 0:
        raise ValueError('len4 and len6 ids collide')
    if np.intersect1d(seqs5.index.values, seqs6.index.values).size > 0:
        print(np.intersect1d(seqs5.index.values, seqs6.index.values))
        raise ValueError('len5 and len6 ids collide')
    #######################################################################

    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('1: At least one thread is contained in multiple sequence' +
                         'dataFrames, which should represent sequences of distinct length')

    all_ids = np.concatenate((all_ids, seqs4.index.values))
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('2: At least one thread is contained in multiple sequence' +
                         'dataFrames, which should represent sequences of distinct length')

    all_ids = np.concatenate((all_ids, seqs5.index.values))
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('3: At least one thread is contained in multiple sequence' +
                         'dataFrames, which should represent sequences of distinct length')

    all_ids = np.concatenate((all_ids, seqs6.index.values))
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('4: At least one thread is contained in multiple sequence' +
                         'dataFrames, which should represent sequences of distinct length')

    all_ids = np.concatenate((all_ids, seqs7.index.values))
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('5: At least one thread is contained in multiple sequence' +
                         'dataFrames, which should represent sequences of distinct length')

    # add empty features to the shorter length sequences to ensure equal dimensionality
    # of data frames, additionally add a 'length' feature to each data frame
    seqs2 = add_empty_feats(seqs2, 's0')
    seqs3 = add_empty_feats(seqs3, 'b1')
    seqs4 = add_empty_feats(seqs4, 's1')
    seqs5 = add_empty_feats(seqs5, 'b2')
    seqs6 = add_empty_feats(seqs6, 's2')
    # set length feature for seqs7
    seqs7['length'] = pd.Series(7, index=seqs7.index)
    # concatenate all sequence data frames, should contain exactly the same columns
    # error checking for the column contents -- ensures all data frames contain
    # the same columns
    same_cols(seqs2, seqs7)
    same_cols(seqs3, seqs7)
    same_cols(seqs4, seqs7)
    same_cols(seqs5, seqs7)
    same_cols(seqs6, seqs7)

    df = pd.concat([seqs2, seqs3, seqs4, seqs5, seqs6, seqs7], sort=False)
    print('Index name: %s' % df.index.name)
    return df


def drop_buyer_thrds(df):
    '''
    Description: Drops all threads where 4 or more offers have been explicitly
    made by the buyer, since these threads violate our assumption that
    both parties can only make 3 offers in all cases. These threads either
    result from data processing errors or our lack of understanding of the
    eBay bargaining procedure.

    Pursue more after MVP has been developed
    '''
    # group by thread length
    df_type = df[['unique_thread_id', 'turn_count', 'offr_type_id']].copy()
    # remove all seller offers from df_type
    df_type = df_type[df_type['offr_type_id'] != 2]
    # count remaining offers for each thread, giving a buyer count for each
    # thread
    byr_cnt = df_type.groupby('unique_thread_id').count()['turn_count']

    # count the toal number of threads before removing any
    tot = len(byr_cnt.index)

    # grab all indices where buyer count is greater than or equal to 4
    # since this should not be possible under our assumptions about the
    # nature of eBay bargaining
    many_byrs = byr_cnt[byr_cnt >= 4].index

    # count the number of threads that will be dropped
    num_drop = len(many_byrs)

    # set df index to unique_thread, turn_count tuple
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

    # drop allthreads with more than 4 buyers
    df.drop(index=many_byrs, level='unique_thread_id', inplace=True)

    # reset index
    df.reset_index(inplace=True, drop=False)

    # give the number of threads dropped due to this processing error
    print('Dropped: %.2f %% of threads due to too many buyers' %
          (num_drop / tot * 100))
    return df


def same_cols(df1, df2):
    '''
    Description: An error checking function that ensures both dataFrames contain the same features
    '''
    cols1 = df1.columns
    cols2 = df2.columns
    infirst = np.setdiff1d(cols1, cols2)
    insecond = np.setdiff1d(cols2, cols1)
    if len(infirst) or len(insecond) != 0:
        print('In the short sequence df')
        print(infirst)
        print('In the long sequence df')
        print(insecond)
        raise ValueError(
            'Columns not correctly duplicated across data frames, missing in 1')


def add_empty_feats(df, final_code, feats=['frac_passed', 'frac_remain', 'passed', 'remain', 'offr', 'date', 'time']):
    '''
    Description: Creates empty offer-level features for sequences
    shorter than the max length sequence and adds a length
    feature giving the length of the this sequence subset
    Inputs:
        df: pandas.DataFrame containing sequences of some certain length
        final_code: offer code of the last offer in the sequence subset
        feats: a list of the offer level features expected for each offer
    '''
    all_offs = all_offr_codes('b3')
    print('Number of offers in the longest sequence: %d' % len(all_offs))
    observed_offs = all_offr_codes(final_code)
    seq_len = 7 - (len(all_offs) - len(observed_offs))
    # create length feature
    df['length'] = pd.Series(seq_len, index=df.index)
    # generate list of offer codes not currently contained in df
    missing_offs = []
    for offr in all_offs:
        if offr not in observed_offs:
            missing_offs.append(offr)
    # iterate over missing offers
    for offr_code in missing_offs:
        # iterate over offer level features
        for offr_feat in feats:
            # generate name for feature corresponding to offr_feat and offr
            featname = '%s_%s' % (offr_feat, offr_code)
            # create a column in the data frame for featname where every
            # entry is np.NaN
            df[featname] = pd.Series(np.NaN, index=df.index)
    # return the resulting df
    return df


def get_last_code(length):
    '''
    Description: Maps an integer giving the length of a sequence
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


def get_ref_cols(df):
    '''
    Description: Copy all the offr_ji columns in the data frame
    to ref_offr_ji so that the original offer values can be walked
    at out of the interpolated values used for the prediction
    '''
    ################################################################
    # Deprecated maybe useful later for sequence length dependent
    # processing
    # final_code = get_last_code(length)
    # seq_cols = all_offr_codes(final_code)
    # longest_cols = all_offr_codes('b3')
    # missing_cols = [code for code in longest_cols if code not in seq_cols]
    ####################################################################
    all_cols = all_offr_codes('b3')
    for code in all_cols:
        curr_offr = 'offr_%s' % code
        ref_offr_name = 'ref_%s' % curr_offr
        df[ref_offr_name] = df[curr_offr]
    return df

###############################################################
# DEPRECATED
# maybe useful later for sequence length dependent processing


def get_ref_cols_wrap(df):
    df = df.groupby('length')
    group_list = []
    for curr_length, group in df:
        print(curr_length)
        new_group = group.copy()
        new_group = get_ref_cols(new_group)
        if new_group is not None:
            group_list.append(new_group)
################################################################


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


def norm_by_recent_offers(df):
    # get all offers in the data set, including start and
    # the last offer (response offer) and remove start price
    all_codes = all_offr_codes('b3')
    all_offrs = ['offr_%s' % code for code in all_codes]

    # iterate over all remaining offers in the list and
    # compute the tuple of reference offers for each (meaning the
    # two most recent offers)
    ref_offrs = [get_ref_offrs(offr_name) for offr_name in all_offrs]

    # create a data frame containing one column for each previous offer
    normed_df = pd.DataFrame(np.NaN, index=df.index, columns=all_offrs)
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
            curr_filled_inds = df[~df[curr_offr].isnull()].index
            normed_df.loc[curr_filled_inds, curr_offr] = (df.loc[curr_filled_inds, curr_offr] /
                                                          df.loc[curr_filled_inds, 'start_price_usd'])
        # otherwise compute the normalized value of the offer --
        # the difference between this offer and the last offer normalized by the
        # difference between the old offer and the previous offer
        # this corresponds to a value of 0 when the current offer is the same as the old
        # offer, ie when the player has not compromised ast all and
        # 1 when the difference is the same as the difference between previous offer
        # old offer, corresopnding to the current player accepting the last player's offer
        else:
            # grab the indices where the current offer isn't np.NaN
            curr_filled_inds = df[~df[curr_offr].isnull()].index
            normed_df.loc[curr_filled_inds, curr_offr] = (
                df.loc[curr_filled_inds, curr_offr]
                - df.loc[curr_filled_inds, old_offr]) / (df.loc[curr_filled_inds, prev_offr]
                                                         - df.loc[curr_filled_inds, old_offr])
        # grab ids of rows with np.nan among the subset of rows we updated
        # -- indicates division of 0 by 0
        # this implies thread where the player made an offer of the same value
        # as the other players most recent counter offer (instead of just accepting
        # it)
        # encode these as 1 for the current offer
        nan_ids = normed_df[normed_df[curr_offr].isna()].index.values
        nan_ids = np.intersect1d(curr_filled_inds.values, nan_ids)

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
    # now drop rows where one of the offr columns equals inf, or -inf
    drop_ids = np.unique(drop_ids)
    print('Num dropped: %d' % len(drop_ids))
    # print(df.loc[drop_ids, ['start_price_usd', 'org_offr_b0', 'org_offr_s0']])
    df.drop(index=drop_ids, inplace=True)
    # return the original data frame, now with normalized offer values
    return df


def clamp_times(df):
    '''
    Description: Ensures all times are positive (inclusive of 0)
    '''
    # grab all sequence codes
    seq_codes = all_offr_codes('b3')
    # iterate over each sequence code
    for code in seq_codes:
        curr_time = 'time_%s' % code
        inds = df[df[curr_time] < 0].index
        df.loc[inds, curr_time] = 0
    return df


def round_inds(df, round_vals):
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
    Output: a pandas data frame containing indicators described above
    '''
    round_vals = [int(round_val) for round_val in round_vals]
    # get list of all offers in the data set except for the next
    # offer (ie response offer)
    offr_set = all_offr_codes('b3')
    offr_set = ['offr_%s' % code for code in offr_set]

    # for debugging
    # print(offer_set)
    # sample_ind = df[df['unique_thread_id'].isin([96, 167, 206])].index
    # print(sample_ind)
    # iterate over round values
    for offr_name in offr_set:
        # get all indices where the current offer isn't undefined
        curr_inds = df[~df[offr_name].isnull()].index
        # grab a subset of the current offer series for the indices
        # which are currently defined
        offr_subset = df.loc[curr_inds, offr_name].copy()
        # iterate over all offers the data set
        # create series to encode whether the offer in question
        # is near but not directly at a round value
        slack_ser = pd.Series(np.NaN, index=df.index)
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
            curr_ser = pd.Series(np.NaN, index=df.index)
            # give a name to the current round series
            new_feat_name = 'rnd_%d_%s' % (curr_val, offr_name)

            # grab a series of the current offer consisting of rows where the
            # offer in this iteration is nonzero because we must divide by this value
            # to determine slack indices -- which would create an NA headache that would
            # ruin my life for sure
            non_zero_offs = offr_subset[offr_subset != 0]
            # debugging
            # print('1:')
            # print('Initial: ')
            # print(non_zero_offs.loc[sample_ind])

            # extract indices for offers that are less than the current value
            below_inds = non_zero_offs[(offr_subset - curr_val) < 0].index

            # find slack indices by finding the remainder of the current offer
            # divided by the current value then dividing this value by the value of the
            # current offer and taking indices where this is below some threshold
            # arbitrary .01 for now
            slack = (non_zero_offs % curr_val) / curr_val

            # print('First slack: ')
            # print(slack.loc[sample_ind])

            # for slack greater than 50 %, subtract from 1 (since this implies the
            # original value is closer to the next factor of the current divisor than
            # the one immediately below it, so it may be in its rounding range, even
            # if not in that of the lower value)
            low_slack_inds = slack[slack > .5].index
            high_slack_inds = slack[slack > .5].index
            slack.loc[high_slack_inds] = 1 - slack.loc[high_slack_inds]
            # grab the intersection of the indices with low slack and the indices for
            # offers worth less than the current value
            low_below_inds = np.intersect1d(
                low_slack_inds.values, below_inds.values)

            # print('Adjusted slack: ')
            # print(slack.loc[sample_ind])
            # truthiness check
            # print(slack.loc[sample_ind] == .01)

            # subset to slack less than .01 of rounding point
            slack = slack[slack <= .011]

            # print('Subset slack')
            # print(sample_ind[sample_ind.isin(slack.index)])
            # print(slack.loc[sample_ind[sample_ind.isin(slack.index)]])

            # separate the indices where slack is 0 and non-zero
            zero_slack = slack[slack == 0].index
            non_zero_slack = slack[slack > 0].index
            # remove low_below_inds from non_zero_slack because these values
            # are in the neighborhood around 0. They're not actually around
            # a round value
            non_zero_slack = np.setdiff1d(
                non_zero_slack.values, low_below_inds)

            # activate the indicator for the non-zero and zero values
            curr_ser.loc[zero_slack] = 1
            curr_ser.loc[non_zero_slack] = 1
            # find the indices where the offer is defined but where the
            # slack is greater than the threshhold, ie the indices where the
            # slack indicator should be defined but 0
            active_inds = np.concatenate(
                (zero_slack.values, non_zero_slack))
            deactive_inds = np.setdiff1d(offr_subset.index.values, active_inds)
            # set the current series to 0 for these indices
            curr_ser.loc[deactive_inds] = 0
            # add the 0 slack indices to a running list of 0 slack indices and
            # the non-zero indices to a running list of non-zero indices for the current offer
            # all_zero_inds = np.append(all_zero_inds, zero_slack.values)
            all_nonzero_inds = np.append(
                all_nonzero_inds, non_zero_slack)
            # finally, add the indicator for the current round-offer pair to the
            # data frame under the name created for it previously
            df[new_feat_name] = curr_ser

        # activate the running list of nonzero slack indices in the slack series
        # for the current offer
        slack_ser.loc[all_nonzero_inds] = 1
        # find all indices in the subset of the df where the current offer is defined
        deactive_inds = np.setdiff1d(offr_subset.index.values, active_inds)
        # deactivate these indices in the slack series
        slack_ser.loc[deactive_inds] = 0
        # finally, add this series to the data frame with the name created for it
        # above
        df[slack_ser_name] = slack_ser
    # return the data frame with the new features
    return df


def get_time_day_feats(df, time_dict):
    '''
    Description: Adds indicator features for each observed offer
    describing whether the offer was received in the morning, afternoon, evening,
    or at night
    Inputs:
        df: pandas.DataFrame containing thread features, including date_ji for
        all observed offers (not including start_price_usd)
        time_dict: dictionary mapping names of periods of time (e.g. morn, eve)
        to len 2 tuples of integers giving the lower and upper bound for the
        times that are considered in the period given by that key...
        Excluding 1 indicator due to linear combination of indicators
    Output: returns a pandas.DataFrame containing 1 indicator feature for each time
    period specified
    '''
    # get all offers in the data set
    offrs = all_offr_codes('b3')

    # iterate over all offers
    for offr in offrs:
        # get a subset of indices where the current date is defined
        curr_inds = df[~df['date_%s' % offr].isnull()].index

        # extract the date feature for this subset of indices
        curr_date = df.loc[curr_inds, 'date_%s' % offr]
        # convert to datetime from string
        curr_date = pd.to_datetime(curr_date)
        # reassign curr date to numpy values
        curr_date = curr_date.values

        # debug check for default type of curr_date so we have correct conversion
        # print(curr_date.dtype)

        # extract the day from each element and convert back to original datetype
        day = curr_date.astype('datetime64[D]').astype(curr_date.dtype)
        # extract the time of day by taking the difference between the
        # observed time and the start of the day
        time_of_day = curr_date - day
        # convert nanoseconds to hours
        time_of_day = time_of_day.astype(int)/1e9/math.pow(60, 2)
        # create empty series with the same ids as the dataframe
        df_time_ser = pd.Series(np.NaN, index=df.index)
        time_of_day = pd.Series(time_of_day, index=curr_inds)
        # iterate over entries in time_dict
        for period, bounds in time_dict.items():
            # extract boundary hours for the current period
            low, high = bounds
            # create series for the current time_period, offer
            # feature pair
            feat_ser = pd.Series(np.NaN, index=df.index)
            # check whether low < high--if not, this index a non-continuous
            # interval where we should consider the interval
            # [low, 24] \lor [0, high]
            if low < high:
                # ids for the rows where the time of day is in
                # the current period
                # in general these are left inclusive intervals
                # so we don't have any collisions
                in_period_ids = time_of_day[(time_of_day >=
                                             low) & (time_of_day < high)].index
            else:
                # ids for the rows where time of day is in the current
                # period for an overnight segment
                in_period_ids = time_of_day[(
                    time_of_day >= low) | (time_of_day < high)].index
            # for all ids in the current period, activate the indicator
            feat_ser.loc[in_period_ids] = 1
            # for all other ids which are currently defined, deactivate the indicator
            deactive_inds = np.setdiff1d(
                curr_inds.values, in_period_ids.values)
            feat_ser.loc[deactive_inds] = 0
            # create a string to name the new feature for the offer,
            # time period pair
            feat_name = '%s_%s' % (period, offr)
            # print(feat_name)
            df[feat_name] = feat_ser
        # finally, add the raw time_of_day as a feature
        # first create a name for the feature
        time_feat_name = 'time_of_day_%s' % offr
        # now add raw times to the df_time_ser at the non-null indices
        df_time_ser.loc[time_of_day.index] = time_of_day.values
        # finally, add this new series to the data frame
        df[time_feat_name] = time_of_day
    return df


def impute_missing_values(df):
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

    ###############################################################
    # setting nans for mssg to 0 arbitrarily since only ~.001% of offers
    # have 0 for message
    # no_msg = df[np.isnan(df['any_mssg'].values)].index
    # df.loc[no_msg, 'any_mssg'] = 0
    # del no_msg
    # BUG MUST ADDRESS THE FACT THAT THIS IS A FEATURE OF THE THREAD NOT
    # OF THE LISTING
    ###############################################################

    return df


def remove_oob(df):
    '''
    Description: Removes normalized offers outside of the [0-1]
    range
    '''
    # get a list of all offers
    offrs = all_offr_codes('b3')
    offrs = ['offr_%s' % code for code in offrs]
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
    Description: This script prepares the data being
    used for the transition probability model training and
    testing. In general, the script iterates over the
    threads in the data, determines how many offers are
    in each thread (not equal to length of the thread
    as its stored in the data set). This corresponds
    to sequence length. Additionally, generates features for
    each offer including time dependent features (how much time
    transpired before the offer was placed, indicators for
    time of day, fraction of auction remaining), round number
    and slack indicators for each offer.


    T B D

    For the time being, stores all the data in three dimensional
    data frame where each row index consists of a tuple giving
    length of the sequence then a subindexer. This architecture
    allows us to avoid creating a length vector for each processed
    chunk. The columns in the data frame
    fully specify all features required for the maximum length
    sequence. We leave columns for offer features of shorter sequences
    undefined (given by 0's)
    '''
    # parser arguments
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    # argument giving filename
    parser.add_argument('--name', action='store', type=str, required=False)
    # argument giving the read/write directory of the file
    parser.add_argument('--dir', action='store', type=str, required=True)
    # argument giving the name of the current experiment
    parser.add_argument('--exp', action='store', type=str, required=True)

    # grab args
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    exp = args.exp

    # error checking for the file being called
    if '_feats2.csv' not in filename:
        raise ValueError('Name of the file should contain _feats2')

    # combine args into read location for current offer csv
    # should be called on some version of '_feats2.csv'
    read_loc = 'data/%s/%s' % (subdir, filename)
    # read chunk
    print('Reading Data')
    df = pd.read_csv(read_loc, parse_dates=['src_cre_date', 'response_time',
                                            'auct_start_dt',
                                            'auct_end_dt'])

    # drop threads which violate our assumptions about the rules eBay sets for
    # bargaining -- investigate more later
    print('Dropping threads with too many buyers')
    df = drop_buyer_thrds(df)

    # convert sets of offer rows into sequences, where each thread corresponds
    # to 1 row and id's correspond to a tuple of (sequence length, thread_id)
    # each thread_id should occur exactly once
    # sequence length is a minimum of 2 (buyer makes an offer, seller responds)
    print('Organizing into sequences')
    df = grab_seqs(df)

    # create a reference column for each offer in the longest sequence
    # Additionally copies columns for shorter sequences to maintain shared
    # dimensionality
    print('Developing reference columns')
    df = get_ref_cols(df)

    # create indicators giving whether each offer is round and whether there is
    # slack at that round point
    print('Generating round offer features')
    df = round_inds(df, [1, 5, 10, 25])

    # normalize all offers by the two most recent offers
    print('Normalizing by recent offers')
    df = norm_by_recent_offers(df)

    # clamp time features to be positive (at least 0 inclusive for automatic
    # replies)
    print('Clamping times')
    df = clamp_times(df)

    # initialize day period dictionary
    time_dict = {
        'morn': (5, 12),
        'aftn': (12, 17),
        'eve': (17, 21),
    }
    # generate additional time features for each offer
    print('Generating additional time features')
    df = get_time_day_feats(df, time_dict)

    # remove threads with out of bounds offers, namely threads where
    # a normalized offer is not in range [0, 1]
    # these may be considered abhorrent threads (in some cases, these
    # result from improper data entry, ie duplicated rows being mistaken as
    # responses from the other party)
    # in most cases, these are a result of "unfaithful" bargaining, abandoning
    # agreement ranging during convergence
    print('Removing offers out of the range of last two offers')
    df = remove_oob(df)

    print('Impute missing values')
    df = impute_missing_values(df)

    # dropping columns that have missing values for the timebeing
    # INCLUDING DROPPING decline, accept prices since it feels
    # epistemologically disingenous to use them
    print('Dropping unnecessary or unusable columns')
    df.drop(columns=['count2', 'count3', 'count4', 'ship_time_fastest', 'ship_time_slowest', 'count1',
                     'ref_price2', 'ref_price3', 'ref_price4', 'decline_price', 'accept_price',
                     'bo_ck_yn', 'item_price', 'anon_item_id', 'anon_thread_id', 'anon_byr_id',
                     'anon_slr_id', 'auct_start_dt', 'auct_end_dt'
                     ], inplace=True)

    # ! FIX
    # drop any_mssg while we wait to address the fact that it is a feature of the offers
    # and not of the the listing itself
    df.drop(columns='any_mssg', inplace=True)

    # drop all dates
    all_codes = all_offr_codes('b3')
    all_dates = ['date_%s' % code for code in all_codes]
    df.drop(columns=all_dates, inplace=True)

    # dropping all threads that do not have ref_price1
    df.drop(df[np.isnan(df['ref_price1'].values)].index, inplace=True)

    # drop outlier threads with extremely high start prices (greater than 1000)
    print('Dropping outliers')
    start_price = df['start_price_usd']
    big_inds = start_price[start_price > 1000].index
    df.drop(index=big_inds, inplace=True)

    # write resulting DataFrame to a csv
    write_path = 'data/exps/%s/%s' % (exp,
                                      filename.replace('_feats2.csv', '.csv'))

    print('Writing output')
    df.to_csv(write_path)


if __name__ == '__main__':
    main()
