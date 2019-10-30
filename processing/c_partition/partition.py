import sys, random
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *


def partition_lstgs(slrs):
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


if __name__ == "__main__":
    # load listing indices
    path = lambda x: FEATS_DIR + '%d_events.gz' % x
    idx = []
    for i in range(1,N_CHUNKS+1):
        idx += list(load(path(i)).index)

    # listings
    lstgs = load(CLEAN_DIR + 'listings.gz')['slr']
    lstgs = lstgs.reindex(index=idx)
    slrs = lstgs.reset_index().sort_values(
        by=['slr','lstg']).set_index('slr').squeeze()
    
    # partition by seller
    partitions = partition_lstgs(slrs)
    dump(partitions, PARTS_DIR + 'partitions.gz')