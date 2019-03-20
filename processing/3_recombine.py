"""
Recombines chunks and splits into dev, test, and train.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = './data/chunks'
SEED = 123456
SHARES = {'dev': .15, 'test': .3}
MODELS = ['delay', 'con', 'round', 'nines', 'msg']
CAT_FEATS = ['meta', 'leaf', 'product', 'cndtn']
OUT_PATH = 'data/simulator/input/'


def save_unique(T):
    u = {}
    for key in CAT_FEATS:
        u[key] = np.unique(T[key])
    pickle.dump(u, open(OUT_PATH + 'cat_feats.pkl', 'wb'))


def save_partitions(slr_dict, x_offer, T, y):
    """
    Saves the data partitions.
    """
    for key, val in slr_dict.items():
        print('Saving', key)
        threads = T.slr.index[np.isin(T.slr.values, val)]
        idx = pd.MultiIndex.from_product([threads, [1, 2, 3]],
            names=x_offer.index.names)
        data = {'x_offer': x_offer.loc[idx],
                'T': T.loc[threads, :],
                'y': {key: val.loc[threads] for key, val in y.items()}}
        path = OUT_PATH + '%s.pkl' % key
        pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # initialize output
    x_offer = pd.DataFrame()
    T = pd.DataFrame()
    idx = pd.MultiIndex.from_product([[], [1, 2, 3]], names=['thread', 'turn'])
    y = {key: pd.Series(index=idx) for key in MODELS}
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
    slr_dict = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        slr_dict[key] = u[last:curr]
        last = curr
    slr_dict['train'] = u[last:]
    return slr_dict


if __name__ == '__main__':
    # append files
    x_offer, T, y = append_chunks()

    # randomize sellers into train, test and pure test
    slr_dict = randomize_sellers(T.slr)

    # partition the data and save
    save_partitions(slr_dict, x_offer, T, y)

    # save vectors of unique values for categorical variables
    save_unique(T)
