"""
Imputes values associated with seller rejections
"""
import argparse
import numpy as np
import pandas as pd
from util_env import extract_datatype


def update_same_vars(df, inds, index, same_vars):
    """
    Passes values from previous index to new entries at this index
    """
    for col in same_vars:
        df.loc[inds, (col, index)] = df.loc[inds, (col, index-1)].values
    return df


def add_rejs(df, index):
    prev = index - 1
    rej_inds = df[df.loc[:, ('item', index - 1)].isin([2, 6, 8])].index
    # compute price of rejection
    if index == 2:
        lstgs = df.loc[rej_inds, ('lstg', 1)]
        price = global_listings.loc[lstgs, 'start_price'].values
    else:
        price = df.loc[rej_inds, ('price', index - 2)].values
    df.loc[rej_inds, ('price', index)] = price
    # compute timestamp of the rejection
    secs = df.loc[rej_inds, ('seconds', index - 1)].values
    clock = df.loc[rej_inds, ('clock', index - 1)].values
    df.loc[rej_inds, ('clock', index)] = secs + clock
    df.loc[rej_inds, ('message', index)] = 0
    # updates thread level variables
    df = update_same_vars(df, rej_inds, index, global_thrd_vars)
    # computes seconds if there is a response
    next_inds = df[~df.loc[:, ('item', index + 1)].isna()].index
    update_inds = np.intersect1d(next_inds.values, rej_inds.values)
    curr_clock = df.loc[update_inds, ('clock', index)].values
    next_clock = df.loc[update_inds, ('clock', index + 1)].values
    df.loc[update_inds, ('seconds', index)] = next_clock - curr_clock
    # compute type
    df.loc[rej_inds('type', index)] = 2
    return df


def impute_rejections(df):
    '''
    Adds rows for seller rejections in each thread

    Args:
        df: dataframe containing standard threads file
    Returns:
        updated dataframe
    '''
    df = df.pivot_table(index='thread', columns='index')
    df = add_rejs(df, 2)
    df = add_rejs(df, 4)
    df = add_rejs(df, 6)
    return df


def impute_acceptances(df):
    """
    Add all acceptances as offers to the data
    """
    # iterate over all offers that might be accepted
    for i in range(1, 7):
        # update price
        accepted_inds = df[df.loc[:, ('status', i)].isin([1, 9])].index
        price = df.loc[accepted_inds, ('price', i)].values
        df.loc[accepted_inds, ('price', i + 1)] = price
        # update clock
        prev_clock = df.loc[accepted_inds, ('clock', i)].values
        prev_secs = df.loc[accepted_inds, ('seconds', i)].values
        df.loc[accepted_inds, ('clock', i + 1)] = prev_clock + prev_secs
        # update vars that are the same
        df = update_same_vars(df, accepted_inds, i + 1, global_thrd_vars)
    return df


def main():
    """
    Main method
    """
    parser = argparse.ArgumentParser()
    # name gives the chunk name (e.g. train-1)
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    offrs_path = 'data/%s/threads/%s_threads.csv' % (datatype, chunk_name)
    listings_path = 'data/%s/listings/%s_listings.csv'
    df = pd.read_pickle(offrs_path)
    # update variables that are the same for each
    global global_thrd_vars
    global_thrd_vars = ['slr_hist', 'byr_hist', 'lstg', 'item',
                        'slr', 'byr', 'thread', 'byr_us', 'bin_rev']
    global global_listings
    global_listings = pd.read_pickle(listings_path)
    global_listings.set_index('lstg', inplace=True, drop=True)
    df = impute_rejections(df)
    df = impute_acceptances(df)
    # pickle dataframe to target
    df.to_pickle(offrs_path)
    # repivot back to normal

    return df
