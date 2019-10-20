import sys, pickle, os, h5py
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
	y = load(getPath(['x', 'thread']))['byr_pctile']

	# initialize fixed features with listing variables
	x_fixed = cat_x_lstg(part).reindex(index=y.index, level='thread')

	# add days since lstg start, holiday, day of week, and minutes since midnight
	threads = load(getPath(['x', 'offer'])).xs(1, level='index')
	cols = ['days', 'holiday', 'hour_of_day'] + \
		[c for c in threads.columns if 'dow' in c]
	x_fixed = x_fixed.join(threads[cols].rename(
		lambda x: 'focal_' + x, axis=1))

	return {'y': y.astype('uint8', copy=False), 
            'x_fixed': x_fixed.astype('float32', copy=False), 
            'x_hour': x_hour.astype('uint16', copy=False)}


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and outcome
	part = PARTITIONS[num]
	print('%s/arrival' % part)

	# out path
    path = lambda x: '%s/%s/%s/arrival.gz' % (PREFIX, x, part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/featnames/con_%s.pkl' % (PREFIX, role), 'wb'))
		pickle.dump(get_sizes(d), 
			open('%s/sizes/con_%s.pkl' % (PREFIX, role), 'wb'))

	# save dictionary of numpy arrays
    dump(convert_to_numpy(d), 
    	'%s/inputs/%s/arrival.gz' % (PREFIX, part))
