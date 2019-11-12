import sys, random
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *


ORDER = ['slr', 'cat', 'title', 'cndtn']


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


# def get_multi_lstgs(L):
#     df = L[LEVELS[:-1] + ['start_date', 'end_time']].set_index(
#         LEVELS[:-1], append=True).reorder_levels(LEVELS).sort_index()
#     # start time
#     df['start_date'] *= 24 * 3600
#     df = df.rename(lambda x: x.split('_')[0], axis=1)
#     # find multi-listings
#     df = df.sort_values(df.index.names[:-1] + ['start'])
#     maxend = df.end.groupby(df.index.names[:-1]).cummax()
#     maxend = maxend.groupby(df.index.names[:-1]).shift(1)
#     overlap = df.start <= maxend
#     return overlap.groupby(df.index.names).max()


if __name__ == "__main__":
    # load listings
    L = load(CLEAN_DIR + 'listings.pkl')

    # find multi-listings
    L = L.sort_values(ORDER + ['start'])
    maxend = L.end.groupby(ORDER).cummax()
    maxend = maxend.groupby(ORDER).shift(1)
    overlap = L.start.astype('int64') * 24 * 3600 <= maxend
    ismulti = overlap.groupby(ORDER + ['lstg']).max()

    # length of listing in days
    days = (L.end_time // (24 * 3600)) - L.start_date + 1

    # drop invalid listings
    L = L.loc[(L.flag == 0) & ~ismulti & (days <= MAX_DAYS)]
    
    # partition by seller
    partitions = partition_lstgs(L.slr)
    dump(partitions, PARTS_DIR + 'partitions.gz')

    # lookup files
    lookup = L[['meta', 'cat', 'start_date', 'end_time', \
        'start_price', 'decline_price', 'accept_price']]
    for part, idx in partitions.items():
        dump(lookup.reindex(index=idx),
                PARTS_DIR + '%s/lookup.gz' % part)
