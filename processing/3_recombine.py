"""
Recombines chunks and splits into pure test, test and train.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = './data/chunks'
SEED = 123456
PCT_PURE = .15
PCT_TEST = .15


def save_partitions(slr_dict, x_offer, T, y):
    """
    Saves the train, test, and pure test partitions.
    """
    for key, val in slr_dict.items():
        print('Saving' + key)
        threads = T.slr.index[np.isin(T.slr.values, val)]
        idx = pd.MultiIndex.from_product([threads, [1, 2, 3]],
            names=x_offer.index.names)
        data = {'x_offer': x_offer.loc[idx, :],
                'T': T.loc[threads, :],
                'y': {key: val.loc[threads] for key, val in y.items()}}
        path = 'data/%s/simulator_input.pkl' % key
        pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # initialize output
    x_offer = pd.DataFrame()
    T = pd.DataFrame()
    idx = pd.MultiIndex.from_product([[], [1, 2, 3]], names=['thread', 'turn'])
    y = {key: pd.Series(index=idx) for key in ['delay', 'con', 'msg']}
    # list of chunks
    chunks = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'simulator' in name]
    # loop over chunks
    for chunk in sorted(chunks):
        chunk = pickle.load(open(chunk, 'rb'))
        x_offer = x_offer.append(chunk['x_offer'], verify_integrity=True)
        T = T.append(chunk['T'], verify_integrity=True)
        for key, val in y.items():
            y[key] = val.append(chunk['y'][key], verify_integrity=True)
    return x_offer, T, y


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
    x_offer, T, y = append_chunks()

    # randomize sellers into train, test and pure test
    slr_dict = randomize_sellers(T.slr)

    # partition the data and save
    save_partitions(slr_dict, x_offer, T, y)
