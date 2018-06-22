import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse
import math
import gc


def grab_turn(df, turn, seller):
    if turn == 0:
        if not seller:
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
            df = df.xs(0, level='turn_count')
            return df
        else:
            print('TBD')
            # fill in seller logic
    elif turn == 1:
        if not seller:
            # count turns in each thread
            thrd_len = df.groupby('unique_thread_id').count()['turn_count']
            # extract thread ids associated with each length thread
            # that can be associated with the second turn
            len_2_ids = thrd_len[thrd_len == 2].index
            len_34_ids = thrd_len[thrd_len].isin([3, 4]).index
            long_ids = thrd_len[thrd_len.isin([5, 6])].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len2 = df.xs(len_2_ids, level='unique_thread_id').copy()
            df.drop(index=len2.index, inplace=True)
            gc.collect()
            len34 = df.xs(len_34_ids, level='unique_thread_id').copy()
            df.drop(index=len34.index, inplace=True)
            gc.collect()
            longdf = df.xs(long_ids,  level='unique_thread_id').copy()

            del df
            del len_2_ids
            del len_34_ids
            del long_ids

            # from length 2 subset, grab threads that contain a seller counteroffer
            seller_counters = len2[len2['offr_type_id'] == 2].index.levels[0]

            gc.collect()

            # now remove all corresponding rows, remaining dataframe only contains threads that
            # correspond to those where the buyer makes two offers & the last row is the
            # buyers second offer
            len2.drop(index=seller_counters,
                      axis='unique_thread_id', inplace=True)
            del seller_counters
            gc.collect()

            # moving on to len34
            # pattern: the buyer's second turn occurs on turn_count = 2 except when a seller
            # counter offer occurs at turn_count = 2

            # remove the fourth offer in each 4 len thread
            len34.drop(index=3, axis='turn_count', inplace=True)

            # extract second turn in each thread
            middle_offrs = len34.xs(1, index='turn_count').copy()
            # thread ids for threads where the middle offer is a seller counter offer
            middle_offr_ids = middle_offrs[middle_offrs['offr_type_id']
                                           == 2].index.levels[0]
            other_offr_ids = middle_offrs[middle_offrs['offr_type_id']
                                          != 2].index.levels[0]
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
            # which necessarily must be
            longdf.drop(index=[1, 3, 4, 5], axis='turn_count', inplace=True)

            # concat all dfs
            out = pd.concat(len2, len34, longdf)
            del len2
            del len34
            del longdf

            # grab all thread ids
            all_threads = out.index.levels[0]
            # create list of tuples correponding to the first offer in each thread
            first_turn_ids = [(thread_id, 0) for thread_id in all_threads]
            # grab the initial offer value using these indices for each thread
            init_offr = out.loc[first_turn_ids, 'offr_price']

            # grab the initial offer time
            init_date = out.loc[first_turn_ids, 'src_cre_date']
            rsp_date = out.loc[first_turn_ids, 'response_time']

            # remove turncount index from the resulting series of offr prices
            init_offr.reset_index(level='turn_count', inplace=True, drop=True)
            init_date.reset_index(level='turn_count', inplace=True, drop=True)
            rsp_date.reset_index(level='turn_count', inplace=True, drop=True)

            # find the difference, in hours, between the response time and the offer time
            diff = (rsp_date.values - init_date.values).astype(int) / \
                1e9/math.pow(60, 2)
            diff = pd.Series(diff, index=rsp_date.index)

            # remove all initial offers
            out.drop(index=first_turn_ids, inplace=True)
            # find ids of all seller offers
            seller_offers = out[out['offr_type_id'] == 2].index
            # remove corresponding rows
            out.drop(index=seller_offers, inplace=True)

            # at this point, there should be only one entry for each thread
            # so we drop the turn_count index
            out.reset_index(level='turn_count', drop=False, inplace=True)
            if (len(np.unique(df.index.values)) != len(df.index)):
                raise ValueError('thread indices are not unique, uh oh')
            # add init_offr series as a new column, both should have the same
            # index, since all threads should still be present
            out['byr_offr_0'] = init_offr
            out['byr_offr_0_resp_time'] = diff
            out.rename(inplace=True, columns={'prev_offr_price': 'slr_offr_0'})
            # add response time for initial offer as a new column
            # df.reset_index(drop=True, inplace=True)
            # matching_rows = df[df['unique_thread_id'].isin(
            #     thread_ids.values)].index
            return out
        else:
            print('TBD')
            # fill in seller logic
    else:
        if not seller:
            # extract second offer df before doing anything else
            off2df = grab_turn(df, 1, seller)
            off2df = off2df[['byr_offr_0', 'offr_price', 'response_time', 'src_cre_date',
                             'offr_0_resp_time', 'prev_offr_price']].copy()
            off2df.rename(columns={'offr_price': 'byr_offr_1'})
            # grab the initial offer time
            init_date = off2df['src_cre_date']
            rsp_date = off2df['response_time']
            # get difference
            diff = (rsp_date.values - init_date.values).astype(int) / \
                1e9/math.pow(60, 2)
            diff = pd.Series(diff, index=off2df.index)
            off2df['byr_offr_1_resp_time'] = diff
            del init_date
            del rsp_date
            off2df.drop(columns=['src_cre_date', 'response_time'])

            thrd_len = df.groupby('unique_thread_id').count()['turn_count']

            len_3_ids = thrd_len[thrd_len == 3].index
            len_4_ids = thrd_len[thrd_len == 4].index
            long_ids = thrd_len[thrd_len.isin([5, 6])].index

            # set index to a multi index of unique
            df.set_index(['unique_thread_id', 'turn_count'], inplace=True)

            # create len specific subsets of df using ids extracted above
            len3 = df.xs(len_3_ids, level='unique_thread_id').copy()
            len4 = df.xs(len_4_ids, level='unique_thread_id').copy()
            longdf = df.xs(long_ids, level='unique_thread_id').copy()

            # len
            del df
            del len_3_ids
            del len_4_ids
            del long_ids

            # focusing on len3--we only want threads where the
            # second and third offers are both buyer offers
            # extracting threads where the second offer is a seller offer
            sec_off = len3.xs(1, level='turn_count',
                              drop=True)['offr_type_id']
            seller_threads = sec_off[sec_off == 2].index

            # remove these threads from the df
            len3.drop(index=seller_threads,
                      axis='unique_thread_id', inplace=True)
            del sec_off
            del seller_threads

            # extracting threads where the third offer is a seller offer
            third_off = len3.xs(index=1, level='turn_count', drop=True)[
                'offr_type_id']
            seller_threads = third_off[third_off == 2].index

            # remove these threads from the df
            len3.drop(index=seller_threads,
                      axis='unique_thread_id', inplace=True)
            del seller_threads
            # from the remaining threads, remove the first and second offers
            len3.drop(index=[0, 1], axis='turn_count', inplace=True)

            # moving on to length 4 threads
            # extracting fourth offer from each thread
            last = len4.xs(3, level='turn_count', drop=True)['offr_type_id']

            # threads where the last offer is made by a seller
            last_seller = last[last['offr_type_id'] == 2].index
            del last

            # threads where the second offer is made by a seller
            two = len4.xs(1, level='turn_count', drop=True)['offr_type_id']
            two_seller = two[two['offr_type_id'] == 2].index

            # from the remaining threads, remove the first and second offers
            len4.drop(index=[0, 1], axis='turn_count', inplace=True)
            del two

            # get threads where both second and last offers are made by the seller
            shared = np.intersect1d(last_seller.values, two_seller.values)
            del two_seller

            # drop all threads where both the second and the last offer
            # are made by the seller
            len4.drop(index=shared, axis='unique_thread_id', inplace=True)
            del shared

            # get all threads where the last offer is made by the seller but
            # the second offer is not (ie remaining threads where second offer
            # is made by the seller)
            # these are the threads where the last buyer offer is located
            # at turn 3 (ie turn_count = 2)
            remaining = len4.index.values[0]
            last_seller = np.intersect1d(remaining, last_seller.values)

            # get all other threads
            remaining = np.setdiff1d(remaining, last_seller)
            remaining = [(thread_id, 3) for thread_id in remaining]
            last_seller = [(thread_id, 2) for thread_id in last_seller]
            all_ids = remaining + last_seller
            del last_seller
            del remaining
            len4 = len4.loc[all_ids].copy()

            # turning attention to longdf
            # all final turns are located in turn_count = 4 (the 5th turn)
            longdf = longdf.xs(index=4, level='turn_count', drop=False).copy()
            out = pd.concat(longdf, len4, len3)
            del longdf
            del len4
            del len3

            # adding features
            out.reset_index(level='turn_count', drop=False, inplace=True)
            out = out.merge(off2df, how='inner',
                            left_index=True, right_index=True)
            del off2df
            out.rename(inplace=True, columns={'prev_offr_price': 'slr_offr_1'})
            return out
        else:
            print('TBD')
            # fill in with seller logic


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
    df = pd.read_csv(path, parse_dates=['src_cre_date', 'response_time'])
    if turn > 2 or turn < 0:
        raise ValueError('Turn must be 0, 1, or 2')

    # calling grab turn method
    df = grab_turn(df, turn, seller)

    type_path = ''
    if seller:
        type_path = 'slr'
    else:
        type_path = 'byr'
    write_path = 'data/' + subdir + '/' + type_path + '_' + \
        str(turn) + '/' + filename.replace('_feats2.csv', '.csv')
    df.to_csv(write_path)


if __name__ == '__main__':
    main()
