"""
-Adds normalized offer price to each offer
-Drops threads where offers converge then diverge

-Adds response values to each offer
    -amount raw dollars
    -delay already given by seconds
    -amount normalized
    -indicator for message
"""
import argparse
import pandas as pd
import numpy as np
from deprecated.util_env import extract_datatype


def add_norm_val(offrs, threads):
    """
    Adds the value of each offer normalized by the previous two
    offers
    """
    # create column giving normalized offer value
    offrs.loc[:, 'norm_price'] = np.NaN
    # iterate through all turns
    for turn in range(1, 8):
        # extract the threads where this turn exists
        curr_threads = offrs.xs(turn, level='index', drop_level=True).index
        # create associated index tuples
        curr_offr_inds = [(ind, turn) for ind in curr_threads.values]
        # if we're normalizing turn 1, just normalize by the start price
        if turn == 1:
            start_price = threads.loc[curr_threads, 'start_price'].values
            offrs.loc[curr_offr_inds, 'norm_price'] = offrs.loc[curr_offr_inds,
                                                                'price'].values / start_price
        else:
            # otherwise extract the recent offers as a numpy array
            rec_offr_inds = [(ind, turn - 1) for ind in curr_threads.values]
            rec_offrs = offrs.loc[rec_offr_inds, 'offr_price'].values
            # if it's turn 2, set the old offers to start price
            if turn == 2:
                old_offrs = threads.loc[curr_threads, 'start_price'].values
            # otherwise set the old offers to 2 turn sago
            else:
                old_offr_inds = [(ind, turn - 2)
                                 for ind in curr_threads.values]
                old_offrs = offrs.loc[old_offr_inds, 'price'].values
            # compute normalized value
            curr_offrs = offrs.loc[curr_offr_inds, 'price'].value
            norm_offrs = (curr_offrs - old_offrs) / (rec_offrs - old_offrs)
            # nan indicates division by 0 (old offer, recent offer, and current offer are all the same)
            norm_offrs[np.isnan(norm_offrs)] = 1
            # update normed price
            offrs.loc[curr_offr_inds, 'norm_price'] = norm_offrs
            # determine which threads have division by 0 (indicates divergence after convergence)
            inf_threads = curr_offr_inds[offrs.loc[curr_offr_inds,
                                                   'norm_price'] == np.inf]
            ninf_threads = curr_offr_inds[offrs.loc[curr_offr_inds,
                                                    'norm_price'] == -np.inf]
            # throw these threads out
            throw_out_threads = np.append(inf_threads, ninf_threads)
            offrs.drop(index=throw_out_threads, level='thread', inplace=True)
            threads.drop(index=throw_out_threads, inplace=True)
    return offrs, threads


def add_message_val(df):
    """
    Adds an indicator for whether each offer
    was answered with a message and adds a field for the
    value of the response offer
    """
    # add new columns
    df.loc[:, 'resp_price'] = np.NaN
    df.loc[:, 'resp_norm_price'] = np.NaN
    df.loc[:, 'resp_msg'] = np.NaN

    # for each turn that might be responded to
    for turn in range(1, 7):
        # compute index of the next turn
        next_turn = turn + 1
        # extract threads where the next turn is defined
        curr_threads = df.xs(next_turn, level='index', drop_level=True).index
        # compute indices
        next_offr_inds = [(ind, next_turn) for ind in curr_threads.values]
        curr_offr_inds = [(ind, turn) for ind in curr_threads.values]
        # extract offers and messages
        next_offrs = df.loc[next_offr_inds, 'price'].values
        next_messages = df.loc[next_offr_inds, 'msg'].values
        # place offers and messages correctly
        df.loc[curr_offr_inds, 'resp_price'] = next_offrs
        df.loc[curr_offr_inds, 'resp_msg'] = next_messages
    return df


def main():
    """
    Main method for parsing required arguments and calling
    functions for adding specific variables
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    # define paths
    offrs_path = 'data/%s/offers/%s_offers.csv' % (datatype, chunk_name)
    lstgs_path = 'data/%s/listings/%s_listings.csv' % (datatype, chunk_name)
    threads_path = 'data/%s/threads/%s_threads.csv' % (datatype, chunk_name)
    # load data
    offrs = pd.read_pickle(offrs_path)
    threads = pd.read_pickle(threads_path)
    lstgs = pd.read_pickle(lstgs_path)

    # rename message column
    offrs.rename({'message': 'msg'}, axis='columns', inplace=True)

    # join threads with lstgs on lstg
    sublstgs = lstgs.loc[:, ['lstg', 'start_price']]
    thread_count = len(threads)
    threads = threads.join(sublstgs, on='lstg', how='inner')
    # when joining, ensure no threads are lost with assert statemen
    assert thread_count == len(threads)
    # create new index for offers and threads
    offrs.set_index(['thread', 'index'], inplace=True, drop=True)
    threads.set_index('thread', inplace=True, drop=True)
    # compute normalize offer value
    offrs, threads = add_norm_val(offrs, threads)
    # add message indicator and value (raw and normalized) of response offer
    offrs = add_message_val(offrs)
    # reset indices and drop column incidently added to threads
    offrs.reset_index(inplace=True)
    threads.reset_index(inplace=True)
    threads.drop(columns='start_price', inplace=True)
    # pickle
    threads.to_pickle(threads_path)
    offrs.to_pickle(offrs_path)


if __name__ == '__main__':
    main()
