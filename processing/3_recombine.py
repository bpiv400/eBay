"""
Recombines chunks and splits into dev, test, and train.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = '../../data/chunks'
SEED = 123456
SHARES = {'dev': .15, 'test': .3}
OUT_PATH = '../../data/simulator/input/'


def save_partitions(slr_dict, O, T, time_feats):
    """
    Saves the data partitions.
    """
    for key, val in slr_dict.items():
        print('Saving', key)
        idx = pd.Index(val, name='lstg')
        # get boolean series of listings specific to seller
        data = {'O': O.loc[idx],
                'T': T.loc[idx].drop('slr', axis=1),
                'time_feats': time_feats.loc[idx]}
        path = OUT_PATH + '%s.pkl' % key
        pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # list of chunks
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'simulator' in name]
    # initialize output
    O = pd.DataFrame()
    T = pd.DataFrame()
    time_feats = pd.DataFrame()
    # loop over chunks
    for path in sorted(paths):
        chunk = pickle.load(open(path, 'rb'))
        O = O.append(chunk['O'], verify_integrity=True)
        T = T.append(chunk['T'], verify_integrity=True)
        time_feats = time_feats.append(chunk['time_feats'],
            verify_integrity=True)
    return O, T, time_feats


def randomize_sellers(u):
    random.seed(SEED)   # set seed
    np.random.shuffle(u)
    slr_dict = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        slr_dict[key] = sorted(u[last:curr])
        last = curr
    slr_dict['train'] = sorted(u[last:])
    return slr_dict


if __name__ == '__main__':
    # append files
    O, T, time_feats = append_chunks()

    # randomize sellers into train, test and dev
    slr_dict = randomize_sellers(
        np.unique(T.index.get_level_values(level='lstg')))

    # partition the data and save
    save_partitions(slr_dict, O, T, time_feats)
