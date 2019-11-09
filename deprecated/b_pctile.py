import sys
from compress_pickle import dump, load
import pandas as pd, numpy as np
from constants import *


# creates series of percentiles indexed by column variable
def get_pctiles(s):
	N = len(s.index)
	s = pd.Series(np.arange(N) / (N-1), 
		index=np.sort(s.values), name='pctile')
	return s.groupby(s.index).min()


# buyer history
T = load(CLEAN_DIR + 'threads.gz')
pctile = get_pctiles(T['byr_hist'])
dump(pctile, '%s/pctile/byr_hist.gz' % PREFIX)

T = T.join(pctile, on='byr_hist').drop(
	'byr_hist', axis=1).rename({'pctile': 'byr_hist'}, axis=1)
dump(T, CLEAN_DIR + 'threads.gz')

# load listings
L = load(CLEAN_DIR + 'listings.gz')

# add arrivals per day to listings
arrivals = T['start_time'].groupby('lstg').count().reindex(
    L.index, fill_value=0)
duration = (L.end_time+1) / (24 * 3600) - L.start_date
L['arrival_rate'] = arrivals / duration

# replace count and rate variables with percentiles
for feat in ['fdbk_score', 'slr_lstgs', 'slr_bos', 'arrival_rate']:
	print(feat)
	pctile = get_pctiles(L[feat])
	if feat != 'arrival_rate':
		dump(pctile, '%s/pctile/%s.gz' % (PREFIX, feat))
	L = L.join(pctile, on=feat)
	L = L.drop(feat, axis=1).rename({'pctile': feat}, axis=1)

# add start_price percentile
pctile = get_pctiles(L['start_price'])
L = L.join(pctile.rename('start_price_pctile'), on='start_price')

# save listings
dump(L, CLEAN_DIR + 'listings.gz')