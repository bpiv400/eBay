"""
Chunks listing, threads, and offers by seller.
"""

import pickle
import pandas as pd, numpy as np

N_CHUNKS = 256

# read in data frames
L = pd.read_csv('data/clean/listings.csv').set_index('lstg')
T = pd.read_csv('data/clean/threads.csv').set_index('thread')
O = pd.read_csv('data/clean/offers.csv').set_index(['thread','index'])

# extract sellers, randomize order, and slice into chunks
sellers = np.unique(L['slr'].values)
step = int(sellers.size / N_CHUNKS) + 1
S = [sellers[m * step:(m+1) * step] for m in range(0, N_CHUNKS)]

# iterate over all chunks
for i in range(1, N_CHUNKS+1):
    # extract associated listings, threads and offers
    L_i = L.join(pd.DataFrame(index=S[i]), on='slr', how='inner')
    T_i = T.join(pd.DataFrame(index=L_i.index), on='lstg', how='inner')
    O_i = O.join(pd.DataFrame(index=T_i.index), how='inner')

    # write chunk
    chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
    path = 'data/chunks/%d.pkl' % i
    pickle.dump(chunk, open(path, 'wb'))