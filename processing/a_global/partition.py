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
    L = load(CLEAN_DIR + 'listings.pkl')

    # find multi-listings
    ismulti = L[['cat', 'title', 'cndtn']].groupby(
        ['cat', 'title']).count().squeeze().rename('ismulti') > 1
    L = L.join(ismulti, on=ismulti.index.names)

    # length of listing in days
    L['days'] = (L.end_time // (24 * 3600)) - L.start_date + 1

    # drop invalid listings
    L = L.loc[~L.flag & ~L.ismulti & (L.days <= MAX_DAYS)]
    L = L.drop(['flag', 'title', 'ismulti', 'days'], axis=1)
    
    # partition by seller
    partitions = partition_lstgs(L.slr)
    dump(partitions, PARTS_DIR + 'partitions.pkl')

    # lookup files
    lookup = L[['meta', 'cat', 'start_date', 'end_time', \
        'start_price', 'decline_price', 'accept_price']]
    for part, idx in partitions.items():
        dump(lookup.reindex(index=idx),
                PARTS_DIR + '%s/lookup.gz' % part)
