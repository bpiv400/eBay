import sys, os, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from processing.processing_consts import *
from constants import *
from featnames import *
from processing.processing_utils import collect_date_clock_feats
from utils import get_months_since_lstg


# function to load file
def load_file(part, x):
	return load(PARTS_DIR + '{}/{}.gz'.format(part, x))


def add_turn_indicators(df):
	'''
	Appends turn indicator variables to offer matrix
	:param df: dataframe with index ['lstg', 'thread', 'index'].
	:return: dataframe with turn indicators appended
	'''
	indices = np.unique(df.index.get_level_values('index'))
	for i in range(len(indices)-1):
		ind = indices[i]
		featname = 't%d' % ((ind+1) // 2)
		df[featname] = df.index.isin([ind], level='index')
	return df


def get_x_thread(threads, idx):
	x_thread = threads.copy()

	# byr_hist as a decimal
	x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

	# reindex to create x_thread
	x_thread = pd.DataFrame(index=idx).join(x_thread)

	# add turn indicators
	if 'index' in threads.index.names:
		x_thread = add_turn_indicators(x_thread)

	return x_thread.astype('float32')


# sets unseen feats to 0
def set_zero_feats(offer, i, outcome):
	# turn number
	turn = offer.index.get_level_values(level='index')

	# all features are zero for future turns
	if i > 1:
		offer.loc[i > turn, :] = 0.0

	# for current turn, set feats to 0
	curr = i == turn
	if outcome == DELAY:
		offer.loc[curr, :] = 0.0
	else:
		offer.loc[curr, MSG] = 0.0
		if outcome == CON:
			offer.loc[curr, [CON, NORM, SPLIT, AUTO, EXP, REJECT]] = 0.0

	return offer


def get_x_offer(offers, idx=None, outcome=None, role=None):
	# initialize dictionary of offer features
	x_offer = {}

	# for threads set role to byr
	if idx is None and outcome is None and role is None:
		role = BYR_PREFIX

	# dataframe of offer features for relevant threads
	if idx is not None:
		threads = idx.droplevel(level='index').unique()
		offers = pd.DataFrame(index=threads).join(offers)

	# last turn to include
	last = max(IDX[role])
	if outcome == 'delay':
		last -= 1

	# turn features
	for i in range(1, last+1):
		# offer features at turn i
		offer = offers.xs(i, level='index').reindex(
			index=idx).astype('float32')

		# set unseen feats to 0 and add turn indicators
		if outcome is not None:
			offer = set_zero_feats(offer, i, outcome)
			offer = add_turn_indicators(offer)

		# set censored time feats to zero
		else:
			if i > 1:
				censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
				offer.loc[censored, TIME_FEATS] = 0.0

		# put in dictionary
		x_offer['offer%d' % i] = offer.astype('float32')

	return x_offer


def init_x(part, idx):
	x = load_file(part, 'x_lstg')
	x = {k: v.reindex(index=idx, level='lstg').astype('float32') for k, v in x.items()}
	return x


def get_arrival_times(lstg_start, lstg_end, thread_start):
	# thread 0: start of listing
	s0 = lstg_start.to_frame().assign(thread=0).set_index(
		'thread', append=True).squeeze()

	# threads 1 to N: real threads
	threads = thread_start.reset_index('thread').drop(
		'clock', axis=1).squeeze().groupby('lstg').max().reindex(
		index=lstg_start.index, fill_value=0)

	# thread N+1: end of lstg
	s1 = lstg_end.to_frame().assign(thread=threads+1).set_index(
		'thread', append=True).squeeze()

	# concatenate and sort into single series
	clock = pd.concat([s0, thread_start, s1], axis=0).sort_index()

	return clock.rename('clock')


def get_interarrival_period(clock):
	# calculate interarrival times in seconds
	df = clock.unstack()
	diff = pd.DataFrame(0.0, index=df.index, columns=df.columns[1:])
	for i in diff.columns:
		diff[i] = df[i] - df[i-1]

	# restack
	diff = diff.rename_axis(clock.index.names[-1], axis=1).stack()

	# original datatype
	diff = diff.astype(clock.dtype)

	# indicator for whether observation is last in lstg
	thread = pd.Series(diff.index.get_level_values(level='thread'),
		index=diff.index)
	last_thread = thread.groupby('lstg').max().reindex(
		index=thread.index, level='lstg')
	censored = thread == last_thread

	# drop interarrivals after BINs
	y = diff[diff > 0]
	censored = censored.reindex(index=y.index)
	diff = diff.reindex(index=y.index)

	# replace censored interarrival times with negative seconds from end
	y.loc[censored] -= MAX_DELAY[ARRIVAL_PREFIX]

	# convert y to periods
	y //= INTERVAL[ARRIVAL_PREFIX]

	return y, diff


def get_x_thread_arrival(clock, idx, lstg_start, diff):
	# seconds since START at beginning of arrival window
	seconds = clock.groupby('lstg').shift().dropna().astype(
		'int64').reindex(index=idx)

	# clock features
	clock_feats = collect_date_clock_feats(seconds)

	# thread count so far
	thread_count = pd.Series(seconds.index.get_level_values(level='thread')-1,
		index=seconds.index, name=THREAD_COUNT)

	# months since lstg start
	months_since_lstg = get_months_since_lstg(lstg_start, seconds)
	assert (months_since_lstg.max() < 1) & (months_since_lstg.min() >= 0)

	# months since last arrival
	months_since_last = diff.groupby('lstg').shift().fillna(0) / MONTH

	# concatenate into dataframe
	x_thread = pd.concat(
		[clock_feats,
		months_since_lstg.rename(MONTHS_SINCE_LSTG),
		months_since_last.rename(MONTHS_SINCE_LAST),
		thread_count], axis=1)

	return x_thread.astype('float32')


def process_arrival_inputs(part, lstg_start, lstg_end, thread_start):
	# arrival times
	clock = get_arrival_times(lstg_start, lstg_end, thread_start)

	# interarrival times
	y, diff = get_interarrival_period(clock)
	idx = y.index

	# listing features
	x = init_x(part, idx)

	# add thread features to x['lstg']
	x_thread = get_x_thread_arrival(clock, idx, lstg_start, diff)
	x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1) 

	return {'y': y, 'x': x}


def save_featnames(x, name):
	'''
	Creates dictionary of input feature names.
	:param x: dictionary of input dataframes.
	:param name: string name of model.
	'''
	# initialize featnames dictionary
	featnames = {k: list(v.columns) for k, v in x.items() if 'offer' not in k}

	# for delay, con, and msg models
	if 'offer1' in x:
		# check that all offer groupings have same organization
		for k in x.keys():
			if 'offer' in k:
				assert all(list(x[k].columns) == OUTCOME_FEATS)

		# one vector of featnames for offer groupings
		featnames['offer'] = list(x['offer1'].columns)

	dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(name))


def save_sizes(x, name):
	'''
	Creates dictionary of input sizes.
	:param x: dictionary of input dataframes.
	:param name: string name of model.
	'''
	sizes = {}

	# count components of x
	sizes['x'] = {k: len(v.columns) for k, v in x.items()}

	# save interval and interval counts
	if (name == 'arrival') or ('delay' in name):
		role = name.split('_')[-1]
		sizes['interval'] = INTERVAL[role]
		sizes['interval_count'] = INTERVAL_COUNTS[role]
		if role == BYR_PREFIX:
			sizes['interval_count_7'] = INTERVAL_COUNTS[BYR_PREFIX + '_7']

		# output size
		sizes['out'] = INTERVAL_COUNTS[role] + 1

	elif name == 'hist':
		sizes['out'] = HIST_QUANTILES
	elif 'con' in name:
		sizes['out'] = CON_MULTIPLIER + 1
	else:
		sizes['out'] = 1

	dump(sizes, INPUT_DIR + 'sizes/{}.pkl'.format(name))


def convert_x_to_numpy(x, idx):
	'''
	Converts dictionary of dataframes to dictionary of numpy arrays.
	:param x: dictionary of input dataframes.
	:param idx: pandas index for error checking indices.
	:return: dictionary of numpy arrays.
	'''
	for k, v in x.items():
		assert np.all(v.index == idx)
		x[k] = v.to_numpy()

	return x


def save_small(d, name):
	# randomly select indices
	v = np.arange(np.shape(d['y'])[0])
	np.random.shuffle(v)
	idx_small = v[:N_SMALL]

	# outcome
	small = {'y': d['y'][idx_small]}

	# inputs
	small['x'] = {k: v[idx_small, :] for k, v in d['x'].items()}

	# save
	dump(small, INPUT_DIR + 'small/{}.gz'.format(name))


# save featnames and sizes
def save_files(d, part, name):
	# featnames and sizes
	if part == 'test_rl':
		save_featnames(d['x'], name)
		save_sizes(d['x'], name)

	# pandas index
	idx = d['y'].index

	# input features
	d['x'] = convert_x_to_numpy(d['x'], idx)

	# convert outcome to numpy
	d['y'] = d['y'].to_numpy()

	# save data
	dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

	# save index
	dump(idx, INDEX_DIR + '{}/{}.gz'.format(part, name))

	# save subset
	if part == 'train_models':
		save_small(d, name)


def save_discrim_files(part, name, x_obs, x_sim):
	# featnames and sizes
	if part == 'train_rl':
		# featnames
		featnames = get_featnames(x_obs)
		dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(name))

		# sizes
		save_sizes(featnames, name)

	# indices
	idx_obs = x_obs['lstg'].index
	idx_sim = x_sim['lstg'].index

	# create dictionary of numpy arrays
	x_obs = convert_x_to_numpy(x_obs, idx_obs)
	x_sim = convert_x_to_numpy(x_sim, idx_sim)

	# y=1 for real data
	y_obs = np.ones(len(idx_obs), dtype=bool)
	y_sim = np.zeros(len(idx_sim), dtype=bool)
	d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

	# join input variables
	assert all(x_obs.keys() == x_sim.keys())
	d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0) for k in x_obs.keys()}

	# save inputs
	dump(d, INPUT_DIR + '{}/listings.gz'.format(part))

	# save joined index
	idx_joined = pd.concat([idx_obs, idx_sim], axis=0)
	dump(idx_joined, INDEX_DIR + '{}/listings.gz'.format(part))

	# save subset
	if part == 'train_rl':
		save_small(d, 'listings')