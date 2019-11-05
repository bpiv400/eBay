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

# creates a single category id from 
def create_category(df):
	# convert categorical variables to strings
	for c in ['meta', 'leaf', 'product']:
		df[c] = np.char.add(c[0], df[c].astype(str).values)
	mask = df['product'] == 'p0'
	df.loc[mask, 'product'] = df.loc[mask, 'leaf']
	# replace infrequent products with leaf
	ct = df['product'].groupby(df['product']).transform('count')
	mask = ct < MIN_COUNT
	df.loc[mask, 'product'] = df.loc[mask, 'leaf']
	df.drop('leaf', axis=1, inplace=True)
	# replace infrequent leafs with meta
	ct = df['product'].groupby(df['product']).transform('count')
	mask = ct < MIN_COUNT
	df.loc[mask, 'product'] = df.loc[mask, 'meta']
	df.drop('meta', axis=1, inplace=True)
	return df.squeeze()


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

# convert to single category
L.loc[:, 'cat'] = create_category(L[['meta', 'product', 'leaf']])
L = L.drop(['meta', 'product', 'leaf'], axis=1)

# save listings
dump(L, CLEAN_DIR + 'listings.gz')