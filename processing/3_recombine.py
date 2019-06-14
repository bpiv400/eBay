"""
Recombines chunks and splits into dev, test, and train.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = '../../data/chunks'
SEED = 123456
SHARES = {'dev': .15, 'test': .3}
OUT_PATH = '../../data/simulator/input/'
TABLE_NAMES = ['time_feats', 'L', 'T', 'O']


def save_partitions(slr_dict, d):
    """
    Saves the data partitions.
    """
    for key, val in slr_dict.items():
        print('Saving', key)
        idx = pd.Index(val, name='slr')
        data = {key: val.loc[idx] for key, val in d.items()}
        path = OUT_PATH + '%s.pkl' % key
        pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # list of chunks
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'out' in name]
    # initialize output
    d = {key: pd.DataFrame() for key in TABLE_NAMES}
    # loop over chunks
    for path in sorted(paths):
        chunk = pickle.load(open(path, 'rb'))
        for key, val in d.items():
            d[key] = val.append(chunk[key], verify_integrity=True)
    return d


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
    d = append_chunks()

    # randomize sellers into train, test and dev
    slrs = np.unique(d['L'].index.get_level_values(level='slr'))
    slr_dict = randomize_sellers(slrs)

    # partition the data and save
    save_partitions(slr_dict, d)
