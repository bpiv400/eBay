import sys
from compress_pickle import load, dump
import pandas as pd, numpy as np
from datetime import datetime as dt
from constants import *


# creates series of percentiles indexed by column variable
def get_pctiles(s):
	N = len(s.index)
	# create series of index name and values pctile
	idx = pd.Index(np.sort(s.values), name=s.name)
	pctiles = pd.Series(np.arange(N) / (N-1), index=idx, name='pctile')
	pctiles = pctiles.groupby(pctiles.index).min()
	# put max in 99th percentile
	pctiles.loc[pctiles == 1] -= 1e-16
	# reformat series with index s.index and values pctiles
	s = s.to_frame().join(pctiles, on=s.name).drop(
		s.name, axis=1).squeeze().rename(s.name)
	return s, pctiles


# count seconds in datetime variable
seconds = lambda t: t.hour*3600 + t.minute*60 + t.second


if __name__ == '__main__':
	# load offers and threads
	O = load(CLEAN_DIR + 'offers.pkl')
	T = load(CLEAN_DIR + 'threads.pkl')

	# convert arrivals to seconds
	s = pd.to_datetime(T.start_time, origin=START, unit='s')

	# split off missing
	toReplace = T.bin & ((T.start_time+1) % (24 * 3600) == 0)
	missing = pd.to_datetime(s.loc[toReplace].dt.date)
	labeled = s.loc[~toReplace]

	# cdf using labeled data
	N = len(labeled.index)
	sec = pd.Series(np.arange(1, N+1) / N, 
		index=np.sort(seconds(labeled.dt).values), name='pctile')
	pctile = sec.groupby(sec.index).min()
	pctile.loc[-1] = 0
	pctile = pctile.sort_index().rename('pctile')

	# calculate censoring second
	last = O.clock.groupby(['lstg', 'thread']).max()
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

	# update thread start time for bins
	T.loc[tdiff.index, 'start_time'] = tdiff

	# byr_hist as percentile
	T.loc[:, 'byr_hist'], toSave = get_pctiles(T['byr_hist'])
	dump(toSave, '%s/pctile/byr_hist.gz' % PREFIX)

	# extract arrivals for later
	arrivals = T['start_time'].groupby('lstg').count()

	# end time of listing
	end_time = tdiff.reset_index(
		'thread', drop=True).rename('end_time')

	# update offers clock in other threads
	df = (O.clock + T['start_time']).reindex(
		index=end_time.index, level='lstg')
	df = df.rename('clock').to_frame().join(end_time)
	idx = df[df['clock'] > df['end_time']].index
	df.loc[idx, 'clock'] = df['end_time']
	df.loc[:, 'clock'] -= T['start_time']
	df.loc[:, 'clock'] = df.clock.astype('int64')
	O.loc[idx, 'clock'] = df.loc[idx, 'clock']
	O.loc[idx, 'censored'] = True

	# save offers and threads
	dump(O, CLEAN_DIR + 'offers.pkl')
	dump(T, CLEAN_DIR + 'threads.pkl')
	del O, T

	# load listings
	L = load(CLEAN_DIR + 'listings.pkl')

	# update listing end time
	L.loc[end_time.index, 'end_time'] = end_time

	# add arrivals per day to listings
	arrivals = arrivals.reindex(index=L.index, fill_value=0)
	duration = (L.end_time+1) / (24 * 3600) - L.start_date
	L['arrival_rate'] = arrivals / duration

	# add start_price percentile
	L['start_price_pctile'], _ = get_pctiles(L['start_price'])

	# convert fdbk_pstv to a rate
	L.loc[:, 'fdbk_pstv'] = L.fdbk_pstv / L.fdbk_score
	L.loc[L.fdbk_pstv.isna(), 'fdbk_pstv'] = 1

	# replace count and rate variables with percentiles
	for feat in ['fdbk_score', 'slr_lstgs', 'slr_bos', 'arrival_rate']:
		print(feat)
		L.loc[:, feat], toSave = get_pctiles(L[feat])
		if feat != 'arrival_rate':
			dump(toSave, '%s/pctile/%s.gz' % (PREFIX, feat))

	# save listings
	dump(L, CLEAN_DIR + 'listings.pkl')
