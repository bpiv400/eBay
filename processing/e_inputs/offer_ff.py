import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *
from processing.e_inputs.inputs import Inputs


def get_x_fixed(idx, x_lstg, x_thread, x_offer, tf, outcome, role):
	# reindex to listings in y
	x_fixed = x_lstg.reindex(index=idx, level='lstg')
	# join with thread features
	x_fixed = x_fixed.join(x_thread[['byr_hist', 'months_since_lstg']])
	# join with raw time features at beginning of thread
	x_fixed = x_fixed.join(tf.xs(0, level='index'))
	# add turn indicators
	x_fixed = add_turn_indicators(x_fixed)
	# merged offer and time feats, excluding index 0
	threads = idx.droplevel(level='index').unique()
	df = pd.DataFrame(index=threads).join(
		x_offer.drop(0, level='index')).join(tf)
	# add in offers
	for i in range(1, max(IDX[role])+1):
		print(i)
		# get features at index i
		offer = df.xs(i, level='index').reindex(index=idx)
		# if turn 1, drop days and delay
		if i == 1:
			offer = offer.drop(['days', 'delay'], axis=1)
		# set features to 0 if i exceeds index
		future = i > offer.index.get_level_values(level='index')
		offer.loc[future, df.dtypes == 'bool'] = False
		offer.loc[future, df.dtypes != 'bool'] = 0
		# for current turn, set feats to 0
		curr = i == offer.index.get_level_values(level='index')
		offer.loc[curr, 'msg'] = False
		if outcome == 'con':
			offer.loc[curr, ['con', 'norm']] = 0
			offer.loc[curr, ['split', 'auto', 'exp', 'reject']] = False
		# if buyer turn, drop irrelevant features
		if i in IDX['byr']:
			offer = offer.drop(['auto', 'exp', 'reject'], axis=1)
		# on last turn, drop irrelevant variables
		if i == max(IDX[role]):
			offer = offer.drop('msg', axis=1)
			if outcome == 'con':
				offer = offer.drop(
					['con', 'norm', 'split', 'auto', 'exp', 'reject'], axis=1)
		# add turn number to feat names and add to x_fixed
		x_fixed = x_fixed.join(
			offer.rename(lambda x: x + '_%d' % i, axis=1))
	return x_fixed


def get_y(x_offer, outcome, role):
	# subset to relevant observations
	if outcome == 'con':
		# drop zero delay and expired offers
		mask = ~x_offer.auto & ~x_offer.exp
	elif outcome == 'msg':
		# drop accepts and rejects
		mask = (x_offer.con > 0) & (x_offer.con < 1)
	s = x_offer.loc[mask, outcome]
	# subset to role
	s = s[s.index.isin(IDX[role], level='index')]
	# for concession, convert to index
	if outcome == 'con':
		s *= 100
		s.loc[(s > 99) & (s < 100)] = 99
		s.loc[(s > 0) & (s < 1)] = 1
		s = np.round(s)
	# convert to byte and return
	return s.astype('int8').sort_index()


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome, role):
	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# load dataframes
	x_lstg = cat_x_lstg(getPath)
	x_thread = load(getPath(['x', 'thread']))
	x_offer = load(getPath(['x', 'offer']))
	tf = load(getPath(['tf', 'role', 'diff'])).reindex(
		index=x_offer.index, fill_value=0)

	# outcome
	y = get_y(x_offer, outcome, role)

	# fixed features
	x_fixed = get_x_fixed(y.index, x_lstg, x_thread, x_offer, tf, outcome, role)

	return {'y': y.astype('int8', copy=False), 
			'x_fixed': x_fixed.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract model from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and role
	part = PARTITIONS[num // 4]
	outcome = 'con' if num % 2 else 'msg'
	role = 'slr' if (num // 2) % 2 else 'byr'
	model = '%s_%s_ff' % (outcome, role)
	print('%s/%s' % (part, model))

	# input dataframes, output processed dataframes
	d = process_inputs(part, outcome, role)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/%s.pkl' % (PREFIX, model), 'wb'))

		pickle.dump(get_sizes(d, model), 
			open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(Inputs(d, model), '%s/inputs/%s/%s.gz' % (PREFIX, part, model))

	# save small dataset
	if part == 'train_models':
		small = create_small(d)
		dump(Inputs(small, model), '%s/inputs/small/%s.gz' % (PREFIX, model))
