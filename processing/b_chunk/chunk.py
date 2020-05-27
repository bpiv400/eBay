import argparse
from compress_pickle import dump, load
from processing.processing_consts import CUTOFF, CLEAN_DIR, \
    CHUNKS_DIR, LVARS, TVARS, OVARS


# split into chunks and save
def create_chunks(group, listings, threads, offers):
    S = listings[group].reset_index().set_index(group).squeeze()
    counts = S.groupby(S.index.name).count()
    groups = []
    total = 0
    num = 1
    for i in range(len(counts)):
        groups.append(counts.index[i])
        total += counts.iloc[i]
        if (i == len(counts) - 1) or (total >= CUTOFF):
            # find corresponding listings
            idx = S.loc[groups]
            # create chunks
            L_i = listings.reindex(index=idx)
            T_i = threads.reindex(index=idx, level='lstg')
            O_i = offers.reindex(index=idx, level='lstg')
            # save
            chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
            path = CHUNKS_DIR + '{}{}.gz'.format(group, num)
            dump(chunk, path)
            # reinitialize
            groups = []
            total = 0
            # increment
            num += 1


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # read in data frames
    listings = load(CLEAN_DIR + 'listings.pkl')
    threads = load(CLEAN_DIR + 'threads.pkl')
    offers = load(CLEAN_DIR + 'offers.pkl')

    # slr features
    if num == 1:
        create_chunks('slr', listings, threads, offers)

    # cat features
    elif num == 2:
        listings = listings[LVARS]
        threads = threads[TVARS]
        offers = offers[OVARS]
        create_chunks('cat', listings, threads, offers)


if __name__ == '__main__':
    main()
