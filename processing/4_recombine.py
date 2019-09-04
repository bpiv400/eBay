"""
Recombines chunks and splits into train, dev, sim, and test.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = '../../data/chunks'
SEED = 123456
SHARES = {'train_models': 1/3, 'train_rl': 1/3}
OUT_PATH = '../../data/partitions/'


def save_partitions(slr_dict, x, y, z):
    for key, val in slr_dict.items():
        print('Saving', key)
        # index of lstgs to slice with
        idx = pd.Index(val, name='lstg')
        # write path
        path = OUT_PATH + '%s.pkl' % key
        # only x_lstg required for simulation
        if key == 'sim':
            pickle.dump(x['lstg'].loc[idx], open(path, 'wb'))
        else:
            data = {}
            data['x'] = {k: v.loc[idx] for k, v in x.items()}
            data['z'] = {k: v.loc[idx] for k, v in z.items()}
            data['y'] = {}
            for model, d in y.items():
                data['y'][model] = {k: v.loc[idx] for k, v in d.items()}
            pickle.dump(data, open(path, 'wb'))


def append_chunks():
    # list of chunks
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'feats' in name]
    # initialize output
    x = {}
    y = {}
    z = {}
    # loop over chunks
    for path in sorted(paths):
        chunk = pickle.load(open(path, 'rb'))
        # x
        for key, val in chunk['x'].items():
            if key in x:
                x[key] = x[key].append(val, verify_integrity=True)
            else:
                x[key] = val
        # y
        for model, d in chunk['y'].items():
            if model not in y:
                y[model] = {}
            for outcome, val in d.items():
                if outcome in y[model]:
                    y[model][outcome] = y[model][outcome].append(
                        val, verify_integrity=True)
                else:
                    y[model][outcome] = val
        # z
        for key, val in chunk['z'].items():
            if key in z:
                z[key] = z[key].append(val, verify_integrity=True)
            else:
                z[key] = val
    return x, y, z


def randomize_lstgs(u):
    random.seed(SEED)   # set seed
    np.random.shuffle(u)
    slr_dict = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        slr_dict[key] = sorted(u[last:curr])
        last = curr
    slr_dict['test'] = sorted(u[last:])
    return slr_dict


if __name__ == '__main__':
    # append files
    x, y, z = append_chunks()

    # randomize sellers into train, test and dev
    lstgs = np.unique(x['lstg'].index.get_level_values(level='lstg'))
    slr_dict = randomize_lstgs(lstgs)

    # partition the data and save
    save_partitions(slr_dict, x, y, z)
