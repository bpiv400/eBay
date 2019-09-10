"""
Recombines chunks and splits into train, dev, sim, and test.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = 'data/chunks'
SEED = 123456
SHARES = {'train_models': 1/3, 'train_rl': 1/3}
OUT_PATH = 'data/partitions/'

X_SUB = ['offer', 'thread', 'lstg']
Z_SUB = ['byr', 'slr', 'start']
ARRIVAL_SUB = ['bin', 'days', 'sec', 'loc', 'hist']
ROLE_SUB = ['msg', 'round', 'delay', 'accept', 'reject', 'nines', 'con']


def partition_lstgs():
    print('Randomly partitioning listings by seller.')
    # list of frames files
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'frames' in name]
    # loop over files, create series of sellers
    flag = False
    for path in sorted(paths):
        chunk = pickle.load(open(path, 'rb'))
        if not flag:
            slrs = chunk['lstgs'].slr
            flag = True
        else:
            slrs = slrs.append(chunk['lstgs'].slr, verify_integrity=True)
    # flip 
    slrs = slrs.reset_index().sort_values(
        by=['slr','lstg']).set_index('slr').squeeze()
    # randomly order sellers
    u = np.unique(slrs.index.values)
    random.seed(SEED)   # set seed
    np.random.shuffle(u)
    # partition listings into dictionary
    d = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        d[key] = np.sort(slrs.loc[u[last:curr]].values)
        last = curr
    d['test'] = np.sort(slrs.loc[u[last:]].values)
    return d


def append_and_partition(paths, partitions, names):
    # append
    print('Appending %s dataframes.' % '_'.join(names))
    flag = True
    for path in sorted(paths):
        # load data
        chunk = pickle.load(open(path, 'rb'))
        # crawl through dictionaries
        for name in names:
            chunk = chunk[name]
        # append to dataframe
        if flag:
            df = chunk
            flag = False
        else:
            df = df.append(chunk, verify_integrity=True)
        del chunk

    # partition
    for key, val in partitions.items():
        filename = key + '/' + '_'.join(names) + '.pkl'
        print('Saving %s' % filename)
        # index of lstgs to slice with
        idx = pd.Index(val, name='lstg')
        # reindex contents of dataframe
        if len(df.index.names) == 1:
            out = df.reindex(index=idx)
        else:
            out = df.reindex(index=idx, level='lstg')
        # write to pickle
        pickle.dump(out, open(OUT_PATH + filename, 'wb'), protocol=4)



if __name__ == '__main__':
    # randomize sellers, create mapping of partition to listings
    partitions = partition_lstgs()

    # list of feats files
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'feats' in name]
    paths = sorted(paths)

    # x
    for sub in X_SUB:
        append_and_partition(paths, partitions, ['x', sub])

    # z
    for sub in Z_SUB:
        append_and_partition(paths, partitions, ['z', sub])

    # y: arrival
    for sub in ARRIVAL_SUB:
        append_and_partition(paths, partitions, ['y', 'arrival', sub])

    # y: role
    for sub in ROLE_SUB:
        for role in ['byr', 'slr']:
            append_and_partition(paths, partitions, ['y', role, sub])
