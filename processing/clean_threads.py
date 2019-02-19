"""
Remove illegal threads from the dataset

Removes threads where:
1. There are offers after an offer has been accepted
2. There are more than 7 observed offers
3. Threads where offer divergence occurs (normalized offer < 0 or > 1)
4. Threads with an offer greater than start price
"""

import argparse
import numpy as np
import pandas as pd
from util_env import extract_datatype


def divergence_threads(offrs):
    """
    Extracts threads where divergence occurs at some point
    (normalized offer price > 1 or < 0)

    Args:
        offrs: df containing offers
    """
    # obtain min normalized offers by thread
    min_offrs = offrs.groupby('thread').min('norm_price')
    # same for max
    max_offrs = offrs.groupby('thread').max('norm_price')
    # extract threads where max offer > 1
    result_max = max_offrs[max_offrs > 1].index.values
    # same for min
    result_min = min_offrs[min_offrs < 0].index.values
    return np.append(result_max, result_min)


def accepted_threads(offrs):
    """
    Extracts threads where offers occur after an offer has been accepted

    Args:
        offrs: df containing offers
    """
    # extract the number of the last turn for each thread
    last_index = offrs.groupby('thread').max('index')
    # convert to dataframe
    last_index = pd.DataFrame(index=last_index.index, columns=['max_index'])
    last_index.index.name = 'thread'
    # subset offers to those where an acceptance occurs
    acc_offrs = offrs.loc[offrs['type'].isin([2]), ['thread', 'index']]
    # join accepted offers to index dataframe
    acc_offrs = acc_offrs.join(last_index, on='thread')
    # extract threads where the max index is greater than the acepted index
    result = acc_offrs.loc[acc_offrs['max_index']
                           > acc_offrs['index'], 'thread'].values
    return result


def high_offr_threads(offrs, threads):
    """
    Extracts threads where the highest offer is greater than 
    the start price

    Args:
        offrs: df giving offer table
        threads: df giving thread ids and start_price

    Returns: numpy array of threads to be removed
    """
    # obtain max offer in each thread
    max_offers = offrs.groupby('thread').max('price')
    # join the threads df with the max_offer series with the thread index
    joined = threads.join(max_offers, on='thread', how='inner')
    # logically index the result by rows where price > start_price
    result = joined.loc[joined['price'] >
                        joined['start_price'], 'thread'].values
    return result


def main():
    """
    Main method for parsing parameters, loading data, and calling functions that handle
    each criteria
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    # define paths to relevant files
    offrs_path = 'data/%s/offers/%s_offers.csv' % (datatype, chunk_name)
    lstgs_path = 'data/%s/listings/%s_listings.csv' % (datatype, chunk_name)
    threads_path = 'data/%s/threads/%s_threads.csv' % (datatype, chunk_name)
    # load data
    offrs = pd.read_pickle(offrs_path)
    threads = pd.read_pickle(threads_path)
    lstgs = pd.read_pickle(lstgs_path)
    # join threads with lstgs on lstg
    sublstgs = lstgs.loc[:, ['lstg', 'start_price']]
    thread_count = len(threads)
    threads_joined = threads.join(sublstgs, on='lstg', how='inner')
    # when joining, ensure no threads are lost with assert statemen
    assert thread_count == len(threads_joined)
    # remove listings and sublstgs
    del lstgs, sublstgs
    # subset joined result to relevant columns
    threads_joined = threads_joined.loc[:, ['thread', 'start_price']]
    # extract threads where an inexplicably high offer has been made
    drop_threads = high_offr_threads(offrs, threads_joined)
    # extract thrads where negotiation diverges
    drop_threads = np.append(drop_threads, divergence_threads(offrs))
    # extrat threads where offers occur after acceptance
    drop_threads = np.append(drop_threads, accepted_threads(offrs))
    drop_threads = np.unique(drop_threads)
    # set indices for threads, drop threads, and reset index
    threads.set_index('thread', inplace=True, drop=True)
    threads.drop(index=drop_threads, inplace=True)
    threads.reset_index(drop=False, inplace=True)
    # set indices for threads, drop threads, and reset index
    offrs.set_index(['thread', 'index'], inplace=True, drop=True)
    offrs.drop(index=drop_threads, inplace=True)
    offrs.reset_index(inplace=True, drop=False)
    # save updated threads and offrs df's
    offrs.to_pickle(offrs_path)
    threads.to_pickle(threads_path)


if __name__ == "__main__":
    main()
