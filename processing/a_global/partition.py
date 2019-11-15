import sys, random
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
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
    # load listings
    L = pd.read_csv(CLEAN_DIR + 'listings.csv', index_col=0)

    # drop flagged listings
    L = L.loc[L.flag == 0]
    L = L.drop('flag', axis=1)
    
    # partition by seller
    partitions = partition_lstgs(L.slr)
    dump(partitions, PARTS_DIR + 'partitions.pkl')

    # lookup files
    lookup = L[['meta', 'cat', 'start_date', 'end_time', \
        'start_price', 'decline_price', 'accept_price']]
    for part, idx in partitions.items():
        dump(lookup.reindex(index=idx),
                PARTS_DIR + '%s/lookup.gz' % part)
