import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# load thread features
	x_thread = load(getPath(['x', 'thread']))
	x_offer = load(getPath(['x', 'offer'])).xs(1, level='index').drop(
		['days', 'delay', 'con', 'norm', 'split', 'msg', 'reject', \
		'auto', 'exp'], axis=1)

	# outcome
	y = x_thread['byr_hist']
	idx = y.index

	# initialize dictionary of input features
	x = init_x(getPath, idx)
	x['lstg'] = x['lstg'].join(x_thread.months_since_lstg).join(x_offer)

	return {'y': y.astype('uint8', copy=False), 
			'x': {k: v.astype('float32', copy=False) for k, v in x.items()}}


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and outcome
	part = PARTITIONS[num]
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
