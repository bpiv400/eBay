"""
Remove illegal threads from the dataset

Removes threads where:
1. There are offers after an offer has been accepted
2. There are more than 7 observed offers
3. Threads where offer divergence occurs (normalized offer < 0 or > 1)
4. Threads with an offer greater than start price
"""

import argparse, pickle
import numpy as np, pandas as pd

def divergence_threads(O):
    """
    Extracts threads where divergence occurs at some point
    (normalized offer price > 1 or < 0)

    Args:
        O: df containing offers
    """
    # obtain min normalized offers by thread
    min_O = O.groupby('thread').min('norm_price')
    # same for max
    max_O = O.groupby('thread').max('norm_price')
    # extract threads where max offer > 1
    result_max = max_O[max_O > 1].index.values
    # same for min
    result_min = min_O[min_O < 0].index.values
    return np.append(result_max, result_min)


def high_offr_threads(O, threads):
    """
    Extracts threads where the highest offer is greater than
    the start price

    Args:
        O: df giving offer table
        threads: df giving thread ids and start_price

    Returns: numpy array of threads to be removed
    """
    # obtain max offer in each thread
    max_offers = O.groupby('thread').max('price')
    # join the threads df with the max_offer series with the thread index
    joined = threads.join(max_offers, on='thread', how='inner')
    # logically index the result by rows where price > start_price
    result = joined.loc[joined['price'] >
                        joined['start_price'], 'thread'].values
    return result

def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    path = 'data/chunks/%d.pkl' % num
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in['listings','threads','offers']]

    # join offers with threads and listings
    O = O.join(T.loc[:,'lstg'], on='thread').join(L, on='lstg')

    # time-varying features

    # discard listings with changed BIN price
    O = O[O.bin_rev == 0].drop('bin_rev', axis=1)

    # discard problematic offers

    # constant features

    # offer sequence



    # extract threads where an inexplicably high offer has been made
    drop_threads = high_offr_threads(O, threads_joined)
    # extract thrads where negotiation diverges
    drop_threads = np.append(drop_threads, divergence_threads(O))

    drop_threads = np.unique(drop_threads)
    # set indices for threads, drop threads, and reset index
    threads.set_index('thread', inplace=True, drop=True)
    threads.drop(index=drop_threads, inplace=True)
    threads.reset_index(drop=False, inplace=True)
    # set indices for threads, drop threads, and reset index
    O.set_index(['thread', 'index'], inplace=True, drop=True)
    O.drop(index=drop_threads, inplace=True)
    O.reset_index(inplace=True, drop=False)
    # save updated threads and O df's
    O.to_pickle(O_path)
    threads.to_pickle(threads_path)


if __name__ == "__main__":
    main()
