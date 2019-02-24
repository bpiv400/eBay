"""
Recombines chunks and splits into pure test, test and train.
"""

import os
import random
import pickle
import torch
import pandas as pd
import numpy as np
from util_env import unpickle

SEED = 123456
PCT_PURE = .20
PCT_TEST = .20


def save_subset(name, slr, included_slrs, offer_feats, const_feats, slr_responses, subset=True):
    """
    Saves a subset of the simulator input components

    Args:
        name: gives the name of the dataset (train, test, pure_test)
        slr: numpy array of size n giving the seller for each thread
        included_slrs: numpy array giving each seller wwho should be included in this subset
        offer_feats: aggregated offer feats of dim n x 3 x k
        const_feats: aggregated const feats of dim n x k2
        slr_responses: aggregated seller responses of dim n x 3 (for now, will update when more responses occur)
    Kwargs:
        subset: boolean giving whether the slr array contains sellers who shouldn't be included in this subset
    """
    # calculate indices of appended datasets to include and subset
    if subset:
        inds = np.isin(slr, included_slrs)
        offer_feats = offer_feats[inds, :, :]
        const_feats = const_feats[inds, :]
        slr_responses = slr_responses[inds, :]
    # combine into dictionary
    dataset = {'offer_feats': offer_feats,
               'const_feats': const_feats,
               'slr_responses': slr_responses
               }
    path = 'data/%s/simulator_input.pkl' % name
    pickle.dump(dataset, open(path, 'wb'))
    # remove the included rows from the dataframes
    if subset:
        return inds
    else:
        pass


def subset_aggregate(inds, slr, offer_feats, const_feats, slr_responses):
    """
    Removes entries from the aggregated inputs which have already been
    included in other datasets given the indices extracted at the previous step
    """
    offer_feats = offer_feats[~inds, :, :]
    slr = slr[~inds]
    const_feats = const_feats[~inds, :]
    slr_responses = slr_responses[~inds, :]
    return slr, offer_feats, const_feats, slr_responses


def main():
    """
    Main method
    """
    # set seed
    random.seed(SEED)
    # generate list of chunks
    directory = './data/chunks'
    chunk_list = ['%s/%s' % (directory, name) for name in os.listdir(
        directory) if os.path.isfile('%s/%s' % (directory, name)) and 'simulator' in name]
    # append chunks
    print(chunk_list)
    offer_feats = []
    const_feats = []
    slr = []
    slr_responses = []
    for chunk in chunk_list:
        chunk = unpickle(chunk)
        offer_feats.append(chunk['offer_feats'])
        const_feats.append(chunk['const_feats'])
        slr.append(chunk['slr'])
        slr_responses.append(chunk['slr_concessions'])
    offer_feats = np.concatenate(offer_feats)
    const_feats = np.concatenate(const_feats)
    slr = np.concatenate(slr)
    slr_responses = np.concatenate(slr_responses)
    # randomly order all components
    indices = np.arange(slr.size)
    np.random.shuffle(indices)
    slr = slr[indices]
    offer_feats = offer_feats[indices, :, :]
    const_feats = const_feats[indices, :]
    slr_responses = slr_responses[indices, :]
    # partition slrs into train, test, and pure_test
    unique_slrs = np.unique(slr)
    num = unique_slrs.size
    test = unique_slrs[:int(num * PCT_TEST)]
    pure_test = unique_slrs[int(num * PCT_TEST)                            : int(num * (PCT_PURE + PCT_TEST))]
    train = unique_slrs[int(num * (PCT_PURE + PCT_TEST)):]
    # save test
    inds = save_subset('test', slr, test, offer_feats,
                       const_feats, slr_responses)
    # update aggregate contents
    slr, offer_feats, const_feats, slr_responses = subset_aggregate(
        inds, slr, offer_feats, const_feats, slr_responses)
    # save pure_test
    inds = save_subset('pure_test', slr, pure_test, offer_feats,
                       const_feats, slr_responses)
    # update aggregate contents
    slr, offer_feats, const_feats, slr_responses = subset_aggregate(
        inds, slr, offer_feats, const_feats, slr_responses)
    # save train
    save_subset('train', slr, train, offer_feats,
                const_feats, slr_responses, subset=False)
    # create dictionary associating seller with dataset for listing code
    slr_dict = {
        'train': train,
        'pure_test': pure_test,
        'test': test
    }
    path = 'data/chunks/slrs.pkl'
    pickle.dump(slr_dict, open(path, 'wb'))


if __name__ == '__main__':
    main()
