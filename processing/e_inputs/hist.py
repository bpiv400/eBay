import sys, pickle, os
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

	# outcome
	y = load(getPath(['x', 'thread'])) * HIST_QUANTILES

	# initialize fixed features with listing variables
	x_fixed = load(getPath(['x', 'lstg'])).reindex(
		index=y.index, level='lstg')

	# load offer dataframe
	offers = load(getPath(['x', 'offer'])).xs(1, level='index')

	# add calendar variables
	toAdd = ['holiday', 'years', 'hour_of_day'] + \
		[c for c in offers.columns if 'dow' in c]
	x_fixed = x_fixed.join(offers[toAdd])

	# add time feats at arrival
	tf = load(getPath(['tf', 'role', 'raw'])).xs(1, level='index')
	x_fixed = x_fixed.join(tf.reindex(index=x_fixed.index, fill_value=0))

	return {'y': y.astype('uint8', copy=False), 
            'x_fixed': x_fixed.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and outcome
	part = PARTITIONS[num]
	print('%s/arrival' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/hist.pkl' % PREFIX, 'wb'))

		pickle.dump(get_sizes(d), 
			open('%s/inputs/sizes/hist.pkl' % PREFIX, 'wb'))

	# save dictionary of numpy arrays
	dump(convert_to_numpy(d), 
		'%s/inputs/%s/hist.gz' % (PREFIX, part))
