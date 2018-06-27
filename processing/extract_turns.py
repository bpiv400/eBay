import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse
import math
import gc


def date_feats(feat_df, col_abv):
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

    return feat_df


# double check what to do about response time and src_cre_date
# for each offer
# ONE SOLUTION
# create encoding for each corresponding to the appropriate offer
# and throw nothing out
# e.g. src_cre_dt_b0, src_cre_dt_s0
# NOTE: Tell Aton about the response time
# TODO: fix all encoding to match the less verbose pattern above

# Downstream
# the dataframe for predicting particpant p turn i shall refered to
# and created under the name of the opposite participant ~p, since the
# prediction is made 'from their point of view

# Additionally, the dataframe contains the date of the offer
# it is seeking to predict as well as the difference in time
# in time between the date of that offer and the previous offer
# We include these variables because we may want to predict them or
# model their distribution at some time in the future based
# on the progression of the thread thusfar, like the price of the
# next offer

# Codebook additions:
# date_bi: date of the buyer's ith offer
# date_si: date of the seller's ith offer
# offr_bi: price of the buyer's ith offer
# offr_si: price of the seller's ith offer
# time_bi: number of hours it took for the buyer to make
# # her ith offer (date_bi - date_s(i-1))
# time_si: number of hours it took for the seller
# to make her ith offer (date_si - date_(b(i-1))
# frac_passed_bi / frac_passed_si: fraction of total auction time passed
# up to bi / si offer (only included when bi is observed, not
# bi is the response offer bieng predicted)
# frac_remain_bi / frac_remain_si: fraction of total auction time remaining
# after bi / si offer (only included when bi is observed, not
# bi is the response offer bieng predicted)
# remain_bi / remain_si: total auction time remaining
# after bi / si offer (only included when bi is observed, not
# bi is the response offer bieng predicted)
# passed_bi / passed_si: total auction time passed
# up to bi / si offer (only included when bi is observed, not
# bi is the response offer bieng predicted)

def get_time(df, end_code, init_code):
    init_date = df['date_' + init_code]
    rsp_date = df['date_' + end_code]

    # find the difference, in hours, between the response time and the offer time
    diff = (rsp_date.values - init_date.values).astype(int) / \
        1e9/math.pow(60, 2)
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
            df = get_time(df, 's0', 'b0')
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
                    off1df = get_time(off1df, 'b1', 's0')
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
            long_ids = thrd_len[thrd_len.isin([5, 6])].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len2 = df.loc[len_2_ids].copy(deep=True)
            df.drop(index=len2.index, inplace=True)
            gc.collect()
            len34 = df.loc[len_34_ids].copy(deep=True)
            df.drop(index=len34.index, inplace=True)
            gc.collect()
            longdf = df.loc[long_ids].copy(deep=True)

            del df
            del len_2_ids
            del len_34_ids
            del long_ids

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

            # moving on to longdf
            # throw out everything past turn_count = 2 and the second turn
            # which necessarily must be a seller offer
            if len(longdf.index) > 0:
                longdf.drop(index=[1, 3, 4, 5],
                            level='turn_count', inplace=True)

            # concat all dfs
            out = pd.concat([len2, len34, longdf], sort=False)
            del len2
            del len34
            del longdf

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
            out = get_time(out, 'b1', 's0')
            out = get_time(out, 's1', 'b1')
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
                len3_comp = get_time(len3_comp, 'b2', 's1')
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
                len4_comp = get_time(len4_comp, 'b2', 's1')
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
            long_ids = thrd_len[thrd_len.isin([5, 6])].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len3 = df.loc[len_3_ids].copy()
            len4 = df.loc[len_4_ids].copy()
            longdf = df.loc[long_ids].copy()

            # len
            del df
            del len_3_ids
            del len_4_ids
            del long_ids

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
                third_off = len3.xs(1, level='turn_count', drop_level=True)[
                    'offr_type_id']
                seller_threads = third_off[third_off == 2].index

                # remove these threads from the df
                len3.drop(index=seller_threads,
                          level='unique_thread_id', inplace=True)
                del seller_threads
                # from the remaining threads, remove the first and second offers
                len3.drop(index=[0, 1], level='turn_count', inplace=True)

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
                # debugging
                id_list = len4.index.values
                # print(id_list[0:20])
                # for ind in all_ids:
                #     if ind not in id_list:
                #         print(ind)
                len4 = len4.loc[all_ids].copy()
                # print(len4)
                # print(len(len4.index))
            # turning attention to longdf
            # all final turns are located in turn_count = 4 (the 5th turn)
            if len(longdf.index) > 0:
                longdf = longdf.xs(4, level='turn_count',
                                   drop_level=False).copy()
            out = pd.concat([longdf, len4, len3])
            del longdf
            del len4
            del len3
            out.rename(columns={'response_time': 'date_s2',
                                'src_cre_date': 'date_b2',
                                'offr_price': 'offr_b2',
                                'resp_offr': 'offr_s2'},
                       inplace=True)
            out.drop(columns=['prev_offr_price', 'status_id',
                              'offr_type_id'], inplace=True)
            # adding features
            out.reset_index(level='turn_count', drop=True, inplace=True)
            out = out.merge(off2df, how='inner',
                            left_index=True, right_index=True)
            del off2df
            out = date_feats(out, 's1')
            out = date_feats(out, 'b2')
            out = get_time(out, 'b2', 's1')
            out = get_time(out, 's2', 'b2')
            out.drop(columns=['passed', 'frac_passed',
                              'remain', 'frac_remain'], inplace=True)
            return out
            # fill in with selle


# at some point, may require a method to extract data useful for predicting initial offers
# from selling price
def get_inits():
    pass


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--dir', action='store', type=str)
    parser.add_argument('--turn', action='store', type=int)
    parser.add_argument('--seller', action='store_true')
    args = parser.parse_args()

    filename = args.name
    turn = args.turn
    seller = args.seller
    subdir = args.dir
    # should be called on feats2
    path = 'data/' + subdir + '/' + filename
    df = pd.read_csv(path, parse_dates=['src_cre_date', 'response_time',
                                        'auct_start_dt',
                                        'auct_end_dt'])
    if turn > 2 or turn < 0:
        raise ValueError('Turn must be 0, 1, or 2')

    # calling grab turn method
    df = grab_turn(df, turn, seller)
    cols = df.columns
    for col in cols:
        if 'time' in col:
            inds = df[df[col] < 0].index
            df.loc[inds, col] = 0

    type_path = ''
    if seller:
        type_path = 's'
    else:
        type_path = 'b'
    write_path = 'data/' + subdir + '/turns/' + \
        type_path + str(turn) + '/' + filename.replace('_feats2.csv', '.csv')
    df.to_csv(write_path)


if __name__ == '__main__':
    main()
