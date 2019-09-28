import sys
sys.path.append('repo/')
import pickle
import pandas as pd, numpy as np
from constants import *


# read in data frames
L = pd.read_csv(CLEAN_DIR + 'listings.csv').set_index('lstg')
T = pd.read_csv(CLEAN_DIR + 'threads.csv').set_index(['lstg', 'thread'])
O = pd.read_csv(CLEAN_DIR + 'offers.csv').set_index(
	['lstg', 'thread','index'])

# unique meta values
u = np.unique(L['meta'])

# iterate over meta
for m in u:
    # extract associated listings and offers
    idx = L.loc[L['meta'] == m].index
    L_i = L.loc[idx, ['meta', 'leaf', 'product', 'start_date', 'end_time']]
    T_i = T.loc[idx, ['start_time']]
    O_i = O.loc[idx, ['clock', 'price', 'accept', 'reject', 'censored']]

    # write chunk
    chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
    path = 'data/chunks/m%d.pkl' % m
    pickle.dump(chunk, open(path, 'wb'))