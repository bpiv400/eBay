"""
Creates chunks of the clean listing and threads dataset
and simultaneously separates the data into train, test, and pure_test
sets

Additionally, creates a unique identifier for each lstg and stores
it as lstg
"""

# packages
import argparse
import random
import pandas as pd
import numpy as np


def type_from_chunk(num, tot_chunks, pure_per, test_per):
    '''
    Computes whether the current chunk should be pure_test, test, or train
    '''
    num_pure = int(tot_chunks * pure_per / 100)
    num_test = int(tot_chunks * test_per / 100)
    if num < num_pure:
        return 'pure_test'
    elif num < (num_pure + num_test):
        return 'test'
    else:
        return 'train'


def main():
    """
    Main method for chunking and folding
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, default=100)
    parser.add_argument('--pure_test', action='store', type=int, default=15)
    parser.add_argument('--test', action='store', type=int, default=20)
    args = parser.parse_args()
    # set kernel
    random.seed(1000)
    # extract thread ids
    threads = pd.read_csv('data/threads_clean.csv')
    listings = pd.read_csv('data/listings_clean.csv')
    # extract listings and randomize the order
    unique_listings = np.unique(listings['item'].values)
    np.random.shuffle(unique_listings)
    # add listings to each chunk
    num_unique = unique_listings.size
    frac = 1 / args.num
    counters = {'pure_test': 1,
                'test': 1,
                'train': 1}

    # iterate over all listings and create a unique listing identifier for each
    # identifier for each slr, lstg pair
    slr = listings[:, 'slr'].values
    lstg = listings[:, 'lstg'].values
    index = pd.MultiIndex.from_arrays([slr, lstg], names=['slr', 'lstg'])
    sldf = pd.Series(data=np.arange(lstg.size), index=index)
    slr = threads['slr'].values
    lstg = threads['lstg'].values
    thrd_ind = list(zip(slr, lstg))
    thrd_lstg = sldf.loc[thrd_ind].values
    threads.loc[:, 'lstg'] = thrd_lstg
    listings.loc[:, 'lstg'] = thrd_lstg

    # iterate over all chunks
    for i in range(args.num):
        # define start and end indices
        start = int(i * num_unique * frac)
        if i < args.num - 1:
            end = int((i + 1) * num_unique * frac)
        else:
            end = num_unique
        items = unique_listings[start:end]
        # extract associated listings and threads
        curr_listings = listings.loc[listings[listings['item'].isin(
            items)].index, :]
        curr_threads = threads.loc[threads[threads['item'].isin(
            items)].index, :]
        # extract associated data type name
        datatype = type_from_chunk(i, args.num, args.pure_test, args.test)
        # define paths
        path_listings = 'data/%s/listings/%s-%d_listings.pkl' % (
            datatype, datatype, counters[datatype])
        path_threads = 'data/%s/threads/%s-%d_threads.pkl' % (
            datatype, datatype, counters[datatype])
        counters[datatype] = counters[datatype] + 1
        # pickle as necessary
        curr_listings.to_pickle(path_listings)
        curr_threads.to_pickle(path_threads)


if __name__ == '__main__':
    main()
