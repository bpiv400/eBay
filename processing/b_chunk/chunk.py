import argparse
from compress_pickle import dump, load
from processing.processing_consts import CUTOFF, CLEAN_DIR, \
    CHUNKS_DIR, LVARS, TVARS, OVARS


# split into chunks and save
def chunk(group, L, T, O):
    S = L[group].reset_index().set_index(group).squeeze()
    counts = S.groupby(S.index.name).count()
    groups = []
    total = 0
    num = 1
    for i in range(len(counts)):
        groups.append(counts.index[i])
        total += counts.iloc[i]
        if (i == len(counts)-1) or (total >= CUTOFF):
            # find correspinding listings
            idx = S.loc[groups]
            # create chunks
            L_i = L.reindex(index=idx)
            T_i = T.reindex(index=idx, level='lstg')
            O_i = O.reindex(index=idx, level='lstg')
            # save
            chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
            path = CHUNKS_DIR + '%s%d.gz' % (group, num)
            dump(chunk, path)
            # reinitialize
            groups = []
            total = 0
            # increment
            num += 1


if __name__ == '__main__':
	# parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # read in data frames
    L = load(CLEAN_DIR + 'listings.pkl')
    T = load(CLEAN_DIR + 'threads.pkl')
    O = load(CLEAN_DIR + 'offers.pkl')

    # slr features
    if num == 1:
    	chunk('slr', L, T, O)

    # cat features
    elif num == 2:
    	L = L[LVARS]
    	T = T[TVARS]
    	O = O[OVARS]
    	chunk('cat', L, T, O)