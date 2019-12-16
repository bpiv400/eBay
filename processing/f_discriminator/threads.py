import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# deletes irrelevant feats and sets unseen feats to 0
def clean_offer(offer, i, outcome, role):
	# if turn 1, drop days and delay
	if i == 1:
		offer = offer.drop(['days', 'delay'], axis=1)
	# set features to 0 if i exceeds index
	else:
		future = i > offer.index.get_level_values(level='index')
		offer.loc[future, offer.dtypes == 'bool'] = False
		offer.loc[future, offer.dtypes != 'bool'] = 0
	# for current turn, set feats to 0
	curr = i == offer.index.get_level_values(level='index')
	if outcome == 'delay':
		offer.loc[curr, offer.dtypes == 'bool'] = False
		offer.loc[curr, offer.dtypes != 'bool'] = 0
	else:
		offer.loc[curr, 'msg'] = False
		if outcome == 'con':
			offer.loc[curr, ['con', 'norm']] = 0
			offer.loc[curr, ['split', 'auto', 'exp', 'reject']] = False
	# if buyer turn or last turn, drop auto, exp, reject
	if (i in IDX['byr']) or (i == max(IDX[role])):
		offer = offer.drop(['auto', 'exp', 'reject'], axis=1)
	# on last turn, drop msg (and concession features)
	if i == max(IDX[role]):
		offer = offer.drop('msg', axis=1)
		if outcome == 'con':
			offer = offer.drop(['con', 'norm', 'split'], axis=1)
	return offer

# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome, role):
	# load dataframes
	x_thread = load(PARTS_DIR + '%s/x_thread.gz' % part)
	x_offer = load(PARTS_DIR + '%s/x_offer.gz' % part)

	# thread indices
	idx = x_thread.index

	# initialize dictionary of input features
	x = init_x(part, idx)
	
	# add thread features to listing features
	x_thread.loc[:, 'byr_hist'] = x_thread.byr_hist.astype('float32') / 10
	x['lstg'] = x['lstg'].join(x_thread)

	# turn indicators
	s = x_offer.reset_index('index')['index']
	s = 

	# offer features
	df = x_offer.unstack(fill_value=0)
	for i in range(1, 8):
		offer = df.xs(i, axis=1, level='index')
		offer = offer.loc[:, offer.std() > 0]
		x['offer' + str(i)] = offer.join(turn_exists)


	

	# turn features
	for i in range(1, 8):
		# offer features at turn i
		offer = df.xs(i, level='index').reindex(index=idx)
		# clean
		offer = clean_offer(offer, i, outcome, role)
		# add turn number to featname
		offer = offer.rename(lambda x: x + '_%d' % i, axis=1)
		# add turn indicators
		x['offer%d' % i] = add_final_turn_indicators(offer)

	# return feature dictionary
	return x


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	args = parser.parse_args()
	part = args.part

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/threads.pkl' % PREFIX, 'wb'))

		pickle.dump(get_sizes(d, model), 
			open('%s/inputs/sizes/threads.pkl' % PREFIX, 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(d, '%s/inputs/%s/threads.gz' % (PREFIX, part))

	# save small dataset
	if part == 'train_models':
		dump(create_small(d), '%s/inputs/small/threads.gz' % PREFIX)
