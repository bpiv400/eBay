from compress_pickle import dump
import numpy as np
from processing.processing_utils import read_csv
from processing.processing_consts import SHARES, SEED
from constants import PARTS_DIR


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
        by=['slr','lstg']).set_index('slr').squeeze()
    # randomly order sellers
    u = np.unique(slrs.index.values)
    np.random.seed(SEED)   # set seed
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
    listings = read_csv('listings')
    
    # partition by seller
    partitions = partition_lstgs(listings.slr)
    dump(partitions, PARTS_DIR + 'partitions.pkl')

    # lookup files
    lookup = listings[['meta',
                       'cat',
                       'start_date',
                       'start_price',
                       'decline_price',
                       'accept_price']].copy()

    lookup.loc[:, 'start_date'] = lookup.start_date.astype('int64') * 24 * 3600
    lookup.rename({'start_date': 'start_time'}, axis=1, inplace=True)
    for part, idx in partitions.items():
        dump(lookup.reindex(index=idx), '{}{}/lookup.gz'.format(PARTS_DIR, part))
