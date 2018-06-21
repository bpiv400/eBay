import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse
import math


def grab_turn(df, turn, seller):
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    if turn == 0:
        if not seller:
            df = df.xs(0, level='turn_count')
            return df
        else:
            print('TBD')
            # fill in seller logic
    elif turn == 1:
        if not seller:
            subset_threads = df.xs(1, level='turn_count', drop=True).index
            df = df.xs(subset_threads.values, level='unique_thread_id')
            df = df.xs([0, 1, 2], level='turn_count')
            # remove turn_count = 2 from threads where the first turn is declined
            # since on these turns turn_count = 1 corresponds to the buyers second turn
            thread_ids = df[df.xs(0, level='turn_count', drop=True)[
                'status_id'].isin([0, 2, 6, 8])].index
            remove_index = [(thread_id, 2) for thread_id in thread_ids.values]
            df.drop(index=remove_index, inplace=True)

            # grab all thread ids
            all_threads = df.index.levels[0]
            # create list of tuples correponding to the first offer in each thread
            first_turn_ids = [(thread_id, 0) for thread_id in all_threads]
            # grab the initial offer value using these indices for each thread
            init_offr = df.loc[first_turn_ids, 'offr_price']

            # grab the initial offer time
            init_date = df.loc[first_turn_ids, 'src_cre_date']
            rsp_date = df.loc[first_turn_ids, 'response_time']

            # remove turncount index from the resulting series of offr prices
            init_offr.reset_index(level='turn_count', inplace=True, drop=True)
            init_date.reset_index(level='turn_count', inplace=True, drop=True)
            rsp_date.reset_index(level='turn_count', inplace=True, drop=True)

            # find the difference, in hours, between the response time and the offer time
            diff = (rsp_date.values - init_date.values).astype(int) / \
                1e9/math.pow(60, 2)
            diff = pd.Series(diff, index=rsp_date.index)

            # remove all initial offers
            df.drop(index=first_turn_ids, inplace=True)
            # find ids of all seller offers
            seller_offers = df[df['offr_type_id'] == 2].index
            # remove corresponding rows
            df.drop(index=seller_offers, inplace=True)

            # at this point, there should be only one entry for each thread
            # so we drop the turn_count index
            df.reset_index(level='turn_count', drop=False, inplace=True)
            if (len(np.unique(df.index.values)) != len(df.index)):
                raise ValueError('thread indices are not unique, uh oh')
            # add init_offr series as a new column, both should have the same
            # index, since all threads should still be present
            df['init_offr'] = init_offr
            df['init_rsp_time'] = diff
            # add response time for initial offer as a new column
            # df.reset_index(drop=True, inplace=True)
            # matching_rows = df[df['unique_thread_id'].isin(
            #     thread_ids.values)].index
            return df
        else:
            print('TBD')
            # fill in seller logic
    else:
        # turn must equal 2 at this point
        # remove all threads that do not contain at least turn_count = 2


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

    df.to_csv(write_path)


if __name__ == '__main__':
    main()
