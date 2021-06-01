import os
import numpy as np
from utils import topickle, load_feats
from constants import FEATS_DIR, PARTS_DIR, NUM_CHUNKS, SHARES
from processing.const import SEED
from featnames import SLR, LSTG, META


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
        by=[SLR, LSTG]).set_index(SLR).squeeze()
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


def create_slr_chunks(listings=None, threads=None, offers=None, chunk_dir=None):
    """
    Chunks data by listing.
    :param DataFrame listings: listing features with index ['lstg']
    :param DataFrame threads: thread features with index ['lstg', 'thread']
    :param DataFrame offers: offer features with index ['lstg', 'thread', 'index']
    :param str chunk_dir: path to output directory
    """
    # split into chunks by seller
    slr = listings[SLR].reset_index().sort_values(
        by=[SLR, LSTG]).set_index(SLR).squeeze()
    u = np.unique(slr.index)
    groups = np.array_split(u, NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        print('Creating slr chunk {} of {}'.format(i + 1, NUM_CHUNKS))
        lstgs = slr.loc[groups[i]].values
        chunk = {'listings': listings.reindex(index=lstgs),
                 'threads': threads.reindex(index=lstgs, level=LSTG),
                 'offers': offers.reindex(index=lstgs, level=LSTG)}
        topickle(chunk, chunk_dir + '{}.pkl'.format(i))


def create_meta_chunks(listings=None, threads=None, offers=None, chunk_dir=None):
    """
    Chunks data by listing.
    :param DataFrame listings: listing features with index ['lstg']
    :param DataFrame threads: thread features with index ['lstg', 'thread']
    :param DataFrame offers: offer features with index ['lstg', 'thread', 'index']
    :param str chunk_dir: path to output directory
    """
    # split into chunks by seller
    meta = listings[META].reset_index().sort_values(
        by=[META, LSTG]).set_index(META).squeeze()
    u = np.unique(meta.index)
    for i in range(len(u)):
        print('Creating meta chunk {} of {}'.format(i + 1, len(u)))
        lstgs = meta.loc[u[i]].values
        chunk = {'listings': listings.reindex(index=lstgs),
                 'threads': threads.reindex(index=lstgs, level=LSTG),
                 'offers': offers.reindex(index=lstgs, level=LSTG)}
        topickle(chunk, chunk_dir + 'meta{}.pkl'.format(i))


def main():
    # load dataframes
    args = {}
    for k in ['listings', 'threads', 'offers']:
        args[k] = load_feats(k)

    # chunk by listing
    chunk_dir = FEATS_DIR + 'chunks/'
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)
    args['chunk_dir'] = chunk_dir
    create_slr_chunks(**args)
    create_meta_chunks(**args)

    # partition by seller
    partitions = partition_lstgs(args['listings'][SLR])
    topickle(partitions, PARTS_DIR + 'partitions.pkl')


if __name__ == '__main__':
    main()
