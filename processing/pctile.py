import sys
from compress_pickle import dump, load
import pandas as pd, numpy as np
from datetime import datetime as dt
from constants import *


# creates series of percentiles indexed by column variable
def get_pctiles(s):
	N = len(s.index)
	s = pd.Series(np.arange(1, N+1) / N, 
		index=np.sort(s.values), name='pctile')
	return s.groupby(s.index).min()

# buyer history
T = load(CLEAN_DIR + 'threads.gz')
pctile = get_pctiles(T['byr_hist'])
dump(pctile, '%s/pctile/byr_hist.gz' % PREFIX)

T = T.join(pctile, on='byr_hist').drop(
	'byr_hist', axis=1).rename({'pctile': 'byr_hist'}, axis=1)
dump(T, CLEAN_DIR + 'threads.gz')

# clean listings
L = load(CLEAN_DIR + 'listings.gz')
L.loc[L.fdbk_score.isna(), 'fdbk_score'] = 0.0
L.loc[:, 'fdbk_score'] = L['fdbk_score'].astype(np.int64)
L['fdbk_pstv'] /= 100
L.loc[L.fdbk_pstv.isna(), 'fdbk_pstv'] = 1

# add arrivals per day to listings
arrivals = T['start_time'].groupby('lstg').count().reindex(
    L.index, fill_value=0)
duration = (L.end_time+1) / (24 * 3600) - L.start_date
L['arrival_rate'] = arrivals / duration
L['arrival_rate'] *= 7

# replace feedback score
for feat in ['fdbk_score', 'slr_lstgs', 'slr_bos', 'start_price']:
	print(feat)
	pctile = get_pctiles(L[feat])
	dump(pctile, '%s/pctile/%s.gz' % (PREFIX, feat))
	L = L.join(pctile, on=feat)
	if feat == 'start_price':
		L = L.rename({'pctile': 'start_price_pctile'}, axis=1)
	else:
		L = L.drop(feat, axis=1).rename({'pctile': feat}, axis=1)

dump(L, CLEAN_DIR + 'listings.gz')