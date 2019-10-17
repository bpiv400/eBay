import sys
sys.path.append('repo/')
from compress_pickle import dump
import pandas as pd, numpy as np
from constants import *

CUTOFF = 1e5


# read in data frames
L = pd.read_csv(CLEAN_DIR + 'listings.csv').set_index('lstg')
T = pd.read_csv(CLEAN_DIR + 'threads.csv').set_index(['lstg', 'thread'])
O = pd.read_csv(CLEAN_DIR + 'offers.csv').set_index(
	['lstg', 'thread','index'])

# assign sellers to chunks
S = L['slr'].reset_index().set_index('slr').squeeze()
counts = S.groupby(S.index.name).count()
sellers = []
total = 0
num = 1
for i in range(len(counts)):
	sellers.append(counts.index[i])
	total += counts.iloc[i]
	if (i == len(counts)-1) or (total >= CUTOFF):
		# find correspinding listings
		idx = S.loc[sellers]
		# create chunks
		L_i = L.reindex(index=idx)
		T_i = T.reindex(index=idx, level='lstg')
		O_i = O.reindex(index=idx, level='lstg')
		# save
		chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
		path = 'data/chunks/%d.gz' % num
		dump(chunk, path)
		# reinitialize
		sellers = []
		total = 0
		# increment
		num += 1
    