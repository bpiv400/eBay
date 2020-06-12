from compress_pickle import dump
import numpy as np
import pandas as pd
from constants import SEED, CLEAN_DIR, PARTS_DIR, TRAIN_MODELS, \
    TRAIN_RL, VALIDATION

# for partitioning
SHARES = {TRAIN_MODELS: 0.6, TRAIN_RL: 0.16, VALIDATION: 0.04}


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


def main():
    # load listings
    listings = pd.read_csv(CLEAN_DIR + 'listings.csv').set_index('lstg')
    
    # partition by seller
    partitions = partition_lstgs(listings.slr)
    dump(partitions, PARTS_DIR + 'partitions.pkl')

    # lookup files
    lookup = listings[['meta',
                       'start_date',
                       'start_price',
                       'decline_price',
                       'accept_price']].copy()

    lookup.loc[:, 'start_date'] = lookup.start_date.astype('int64') * 24 * 3600
    lookup.rename({'start_date': 'start_time'}, axis=1, inplace=True)
    for part, idx in partitions.items():
        dump(lookup.reindex(index=idx), '{}{}/lookup.gz'.format(PARTS_DIR, part))


if __name__ == "__main__":
    main()
