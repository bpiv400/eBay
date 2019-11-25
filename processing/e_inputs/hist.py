import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *
from processing.e_inputs.inputs import Inputs


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

	# initialize fixed features with listing variables
	x_fixed = cat_x_lstg(getPath).reindex(index=y.index, level='lstg')

	# add thread variables
	x_fixed = x_fixed.join(x_thread.months_since_lstg)
	x_fixed = x_fixed.join(x_offer)

	return {'y': y.astype('uint8', copy=False), 
            'x_fixed': x_fixed.astype('float32', copy=False)}


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
	dump(Inputs(d, 'hist'), '%s/inputs/%s/hist.gz' % (PREFIX, part))

	# save small dataset
	if part == 'train_models':
		small = create_small(d)
		dump(Inputs(small, 'hist'), '%s/inputs/small/hist.gz' % PREFIX)
