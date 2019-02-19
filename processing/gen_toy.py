"""
Script for generating a toy dataset
"""
import random
import pandas as pd
import numpy as np


def main():
    """
    Main method which loads relevant data files and subsets them
    """
    # paths
    gen_path = 'data/train/'
    offers_path = '%soffers/train-25_offers.pkl'
    lstg_path = '%slistings/train-25_listings.pkl'
    threads_path = '%sthreads/train-25_threads.pkl'
    # load files
    lstgs = pd.read_pickle(lstg_path)
    threads = pd.read_pickle(threads_path)
    offers = pd.read_pickle(offers_path)

    # grab 10% of threads
    random.seed(100)
    toy_threads = threads.sample(10)
    # grab listings associated with those threads
    toy_lstgs = lstgs.loc[lstgs['lstg'].isin(
        np.unique(toy_threads['lstg'].values)), :]
    # grab offers associated with those threads
    toy_offers = offers.loc[offers['thread'].isin(
        toy_threads['thread'].values), :]
    # target directory
    targ_dir = 'data/toy/'
    # pickle each file
    toy_lstgs.to_pickle('%slistings/toy-1_listings.pkl')
    toy_lstgs.to_pickle('%sthreads/toy-1_threads.pkl')
    toy_lstgs.to_pickle('%soffers/toy-1_offers.pkl')

    # some unrelated assertions about data integrity
    # check that all threads in offers are in threads
    assert offers['thread'].isin(threads['thread']).all()
    # check that all listings in thread are contained in lstgs
    assert threads['lstg'].isin(lstgs['lstg']).all()
    # check that lstg is unique in lstg
    assert lstg['lstg'].size == lstg['lstg'].unique().size
    # check that thread is unique in thread
    assert threads['thread'].size == threads['thread'].unique().size


if __name__ == "__main__":
    main()
