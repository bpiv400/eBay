import sys
sys.path.append('repo/')
import pickle
import pandas as pd, numpy as np
from constants import *

# read in data frames
L = pd.read_csv(CLEAN_DIR + 'listings.csv').set_index('lstg')
O = pd.read_csv(CLEAN_DIR + 'offers.csv').set_index(
	['lstg', 'thread','index'])

# unique meta values
u = np.unique(L['meta'])

# iterate over meta
for m in u:
    # extract associated listings and offers
    idx = L.loc[L['meta'] == m].index
    L_i = L.loc[idx]
    O_i = O.loc[idx]

    # write chunk
    chunk = {'listings': L_i, 'offers': O_i}
    path = 'data/chunks/m%d.pkl' % m
    pickle.dump(chunk, open(path, 'wb'))