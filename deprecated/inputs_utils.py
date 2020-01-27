import sys, os
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from processing.processing_consts import *
from constants import *
from featnames import *


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
		feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS + TURN_FEATS[name]

		# check that all offer groupings have same organization
		for k in x.keys():
			if 'offer' in k:
				assert list(x[k].columns) == feats
					
		# one vector of featnames for offer groupings
		featnames['offer'] = feats

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
