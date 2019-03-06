"""
Recombines chunks and splits into pure test, test and train.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = './data/chunks'
SEED = 123456
PCT_PURE = .15
PCT_TEST = .15


def save_partitions(slr_dict, slr, x_offer, x_fixed, y):
    """
    Saves the train, test, and pure test partitions.
    """
    for key in slr_dict:
        threads = slr.index[np.isin(slr.values, slr_dict[key])]
        idx = pd.MultiIndex.from_product([threads, [1, 2, 3]],
            names=x_offer.index.names)
        data = {'x_offer': x_offer.loc[idx, :],
                'x_fixed': x_fixed.loc[threads, :],
                'y': y.loc[threads],
                'slr': slr.loc[threads]}
        path = 'data/%s/simulator_input.pkl' % key
        pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # initialize output
    x_offer = pd.DataFrame()
    x_fixed = pd.DataFrame()
    y = pd.DataFrame()
    slr = pd.Series()
    # list of chunks
    chunks = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'simulator' in name]
    # loop over chunks
    for chunk in sorted(chunks):
        chunk = pickle.load(open(chunk, 'rb'))
        x_offer = x_offer.append(chunk['x_offer'], verify_integrity=True)
        x_fixed = x_fixed.append(chunk['x_fixed'], verify_integrity=True)
        y = y.append(chunk['y'], verify_integrity=True)
        slr = slr.append(chunk['slr'], verify_integrity=True)
    return x_offer, x_fixed, y, slr


def randomize_sellers(slr):
    u = np.unique(slr.values)
    random.seed(SEED)   # set seed
    np.random.shuffle(u)
    test = u[:int(u.size * PCT_TEST)]
    pure = u[int(u.size * PCT_TEST): int(u.size * (PCT_PURE + PCT_TEST))]
    train = u[int(u.size * (PCT_PURE + PCT_TEST)):]
    return {'train': np.sort(train), 'pure': np.sort(pure), 'test': np.sort(test)}


if __name__ == '__main__':
    # append files
    x_offer, x_fixed, y, slr = append_chunks()

    # randomize sellers into train, test and pure test
    slr_dict = randomize_sellers(slr)

    # partition the data and save
    save_partitions(slr_dict, slr, x_offer, x_fixed, y)
