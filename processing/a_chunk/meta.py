import sys
sys.path.append('repo/')
from compress_pickle import dump, load
import pandas as pd, numpy as np
from constants import *

LVARS = ['meta', 'leaf', 'cndtn', 'start_date', 'end_time', \
    'start_price', 'arrival_rate', 'flag']
TVARS = ['start_time', 'byr_hist', 'bin']
OVARS = ['clock', 'price', 'accept', 'reject', 'censored']


# read in data frames
L = load(CLEAN_DIR + 'listings.gz')[LVARS]
T = load(CLEAN_DIR + 'threads.gz')[TVARS]
O = load(CLEAN_DIR + 'offers.gz')[OVARS]

# unique meta values
u = np.unique(L['meta'])

# iterate over meta
for num in u:
    # extract associated listings and offers
    idx = L.loc[L['meta'] == num].index
    L_i = L.reindex(index=idx)
    T_i = T.reindex(index=idx, level='lstg')
    O_i = O.reindex(index=idx, level='lstg')

    # write chunk
    chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
    path = 'data/chunks/m%d.gz' % num
    dump(chunk, path)