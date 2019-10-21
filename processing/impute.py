import sys
sys.path.append('repo/')
from compress_pickle import dump
import pandas as pd, numpy as np
from datetime import datetime as dt
from constants import *

# offers dataframe
O = pd.read_csv(CLEAN_DIR + 'offers.csv').set_index(
	['lstg', 'thread','index'])

# save offers and delete
clock = O.clock.copy()
dump(O, CLEAN_DIR + 'offers.gz')
del O

# seconds function
seconds = lambda t: t.hour*3600 + t.minute*60 + t.second

# threads dataframe
T = pd.read_csv(CLEAN_DIR + 'threads.csv').set_index(
	['lstg', 'thread'])

# convert arrivals to seconds
s = pd.to_datetime(T.start_time, origin=START, unit='s')

# split off missing
toReplace = T.bin & ((T.start_time+1) % (24 * 3600) == 0)
missing = pd.to_datetime(s.loc[toReplace].dt.date)
labeled = s.loc[~toReplace]

# cdf using labeled data
sec = seconds(labeled.dt)
pctile = (sec.groupby(sec).count() / sec.count()).cumsum()
pctile.loc[-1] = 0
pctile = pctile.sort_index().rename('pctile')

# calculate censoring second
last = clock.groupby(['lstg', 'thread']).max()
last = last.loc[~toReplace] + T.start_time
last = last.groupby('lstg').max()
last = pd.to_datetime(last, origin=START, unit='s')
last = last.reindex(index=missing.index, level='lstg').dropna()

# restrict censoring seconds to same day as bin with missing time
sameDate = pd.to_datetime(last.dt.date) == missing.reindex(last.index)
lower = seconds(last.dt).loc[sameDate].rename('lower')
lower = lower.loc[lower < 3600 * 24 - 1]
tau = lower.to_frame().join(pctile, on='lower')['pctile']
tau = tau.reindex(index=missing.index, fill_value=0)

# uniform random
rand = pd.Series(np.random.rand(len(missing.index)), 
	index=missing.index, name='x')

# amend rand for censored observations
rand = tau + (1-tau) * rand

# read off of cdf
cdf = pctile.rename('x').reset_index().set_index('x').squeeze()
newsec = cdf.reindex(index=rand, method='ffill').values
delta = pd.Series(pd.to_timedelta(newsec, unit='second'), 
	index=rand.index)

# new bin arrival times
tdiff = (missing + delta - pd.to_datetime(START))
tdiff = tdiff.dt.total_seconds().astype('int64')

# update thread start time
T.loc[missing.index, 'start_time'] = tdiff

# percentiles of buyer history
idx = np.arange(101)
hist_pct = pd.Series(idx, 
	index=np.percentile(T.byr_hist.values, idx).astype('int32'))
hist_pct = hist_pct.groupby(hist_pct.index).first()
byr_pctile = hist_pct.reindex(
	index=T.byr_hist, method='ffill').values
byr_pctile.loc[byr_pctile == 100] = 99	# only max is in the 100th pctile

# save unique byr percentiles
u = np.unique(byr_pctile)
dump(u, CLEAN_DIR + 'byr_pctiles.gz')

# save threads
T['byr_pctile'] = byr_pctile
dump(T, CLEAN_DIR + 'threads.gz')
del T

# update listing end time
tdiff.reset_index('thread', drop=True, inplace=True)
L = pd.read_csv(CLEAN_DIR + 'listings.csv').set_index('lstg')
L.loc[tdiff.index, 'end_time'] = tdiff

# save listings
dump(L, CLEAN_DIR + 'listings.gz')
