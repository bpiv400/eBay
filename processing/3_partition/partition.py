import sys, random
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *


def partition_lstgs(lstgs):
    slrs = lstgs['slr'].reset_index().sort_values(
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


if __name__ == "__main__":
    # load listing indices
    path = lambda x: FEATS_DIR + str(x) + '_tf_slr.gz'
    idx = []
    for i in range(1,N_CHUNKS+1):
        idx += list(load(path(i)).index)

    # listings
    lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv', 
        index_col='lstg', usecols=['lstg', 'slr'])
    lstgs = lstgs.reindex(index=idx)
    
    # partition by seller
    partitions = partition_lstgs(lstgs)
    dump(partitions, PARTS_DIR + 'partitions.gz')