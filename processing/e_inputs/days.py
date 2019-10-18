import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

from constants import *
from utils import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
	# path name function
	getPath = lambda names: \
		'data/partitions/%s/%s.gz' % (part, '_'.join(names))

	# outcome
	y = load(getPath(['y_arrival_days']))

	# fixed features
	x_fixed = cat_x_lstg(part)

	# day features to merge in collate function
	x_days = pd.DataFrame(index=y.index)
	x_days['days'] = y.index.get_level_values('period')
	clock = pd.to_datetime(x_fixed.start_days + x_days.days, 
        unit='D', origin=START)
	x_days = x_days.join(extract_day_feats(clock))

	return y, x_fixed, x_days


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)

	# partition and outcome
	part = PARTITIONS[parser.parse_args().num-1]
	outfile = lambda x: 'data/inputs/part/arrival_days.pkl' % x
	print('Partition: %s' % part)

	# input dataframes, output processed dataframes
	y, x_fixed, x_days = process_inputs(part)

	# save featnames and sizes once
	if part == 'train_models':
		# save featnames
		featnames = {'x_fixed': x_fixed.columns + x_days.columns}
		pickle.dump(featnames, open(outfile('featnames'), 'wb'))

		# get data size parameters and save
		sizes = {'N': len(x_fixed.index), 
				 'fixed': len(x_fixed.columns) + len(x_days.columns),
				 'out': 2}
		pickle.dump(sizes, open(outfile('sizes'), 'wb'))

	# convert to numpy arrays, save in hdf5
	path = 'data/inputs/%s/arrival_days.hdf5' % part
	f = h5py.File(path, 'w')
	for var in ['y', 'x_fixed', 'x_days']:
		array = globals()[var].to_numpy().astype('float32')
		f.create_dataset(var, data=array, dtype='float32')
	f.close()