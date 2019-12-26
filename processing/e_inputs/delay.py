import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import clean_offer, add_turn_indicators, split_files





# loads data and calls helper functions to construct train inputs
def process_inputs(part, role):
	# function to load file
	load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

	# load dataframes
	x_offer = load_file('x_offer')
	x_thread = load_file('x_thread')

	# outcome
	y = load_file('y_delay_{}'.format(role))

	# initialize dictionary of input features
	x = load_file('x_lstg')
	x = {k: v.reindex(index=idx, level='lstg') for k, v in x.items()}

	# add thread features and turn indicators to listing features
	x_thread.loc[:, 'byr_hist'] = x_thread.byr_hist.astype('float32') / 10
	x['lstg'] = x['lstg'].join(x_thread)
	x['lstg'] = add_turn_indicators(x['lstg'])

	# dataframe of offer features for relevant threads
	threads = idx.droplevel(level='index').unique()
	df = pd.DataFrame(index=threads).join(x_offer)

	# turn features
	for i in range(1, max(IDX[role])+1):
		# offer features at turn i
		offer = df.xs(i, level='index').reindex(index=idx)
		# clean
		offer = clean_offer(offer, i, 'delay', role)
		# add turn number to featname
		offer = offer.rename(lambda x: x + '_%d' % i, axis=1)
		# add turn indicators
		x['offer%d' % i] = add_turn_indicators(offer)

	# index of first x_clock for each y
	idx_clock = load_file('clock').groupby(
		['lstg', 'thread']).shift().reindex(index=idx).astype('int64')

	# normalized periods remaining at start of delay period
	lstg_start = load_file('lookup').start_time
	remaining = MAX_DAYS * 24 * 3600 - (delay_start - lstg_start)
	remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
		MAX_DELAY['slr']
	remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
		MAX_DELAY['byr']
	remaining = np.minimum(remaining, 1)

	# time features
	tf = load_file('tf_delay_{}'.format(role))

	return {'y': y.astype('int8', copy=False), 'x': x,
			'idx_clock': idx_clock.astype('int64', copy=False),
			'remaining': remaining.astype('float32', copy=False),
			'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--role', type=str)
	args = parser.parse_args()
	part, role = args.part, args.role
	name = 'delay_' + role
	print('%s/%s' % (part, name))

	# input dataframes, output processed dataframes
	d = process_inputs(part, role)

	# save various output files
	save_files(d, part, name)
