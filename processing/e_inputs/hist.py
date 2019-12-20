import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition, init_x
from processing.processing_utils import convert_to_numpy, get_featnames, get_sizes, create_small


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	# path name function
	getPath = lambda names: PARTS_DIR + '%s/%s.gz' % \
		(part, '_'.join(names))

	# thread features
	x_offer = load(getPath(['x', 'offer'])).xs(1, level='index')
	x_offer = x_offer.drop(['days', 'delay', 'con', 'norm', 'split', \
		'msg', 'reject', 'auto', 'exp'], axis=1)
	x_offer = x_offer.rename(lambda x: x + '_1', axis=1)
	x_thread = load(getPath(['x', 'thread'])).join(x_offer)

	# outcome
	y = x_thread['byr_hist']
	idx = y.index
	x_thread.drop('byr_hist', axis=1, inplace=True)

	# initialize input features
	x = init_x(part, idx)

	# add thread variables to x['lstg']
	x['lstg'] = x['lstg'].join(x_thread)

	return {'y': y.astype('uint8', copy=False), 
			'x': {k: v.astype('float32', copy=False) for k, v in x.items()}}


if __name__ == '__main__':
	# partition name from command line
	part = input_partition()
	print('%s/hist' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/hist.pkl' % PREFIX, 'wb'))

		pickle.dump(get_sizes(d, 'hist'), 
			open('%s/inputs/sizes/hist.pkl' % PREFIX, 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(d, '%s/inputs/%s/hist.gz' % (PREFIX, part))

	# save small dataset
	if part == 'train_models':
		dump(create_small(d), '%s/inputs/small/hist.gz' % PREFIX)
