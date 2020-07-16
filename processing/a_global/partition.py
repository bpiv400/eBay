from compress_pickle import dump
import numpy as np
import pandas as pd
from constants import SEED, FEATS_DIR, MODEL_PARTS_DIR, SHARES


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
        by=['slr', 'lstg']).set_index('slr').squeeze()
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
    listings = pd.read_csv(FEATS_DIR + 'listings.csv').set_index('lstg')
    
    # partition by seller
    partitions = partition_lstgs(listings.slr)
    dump(partitions, MODEL_PARTS_DIR + 'partitions.pkl')


if __name__ == "__main__":
    main()
