import sys, pickle, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.e_inputs.inputs_util import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part):
	# outcome
	y = load('data/partitions/%s/y_arrival_days.gz' % part)

	# fixed features
	x_fixed = cat_x_lstg(part)

	# day features to merge in collate function
	x_days = pd.DataFrame(index=y.index)
	x_days['days'] = y.index.get_level_values('period')
	clock = pd.to_datetime(x_fixed.start_days + x_days.days, 
        unit='D', origin=START)
	x_days = x_days.join(extract_day_feats(clock))

	return {'y': y.astype('uint8', copy=False), 
			'x_fixed': x_fixed.astype('float32', copy=False), 
			'x_days': x_days.astype('uint8', copy=False)}


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)

	# partition and outcome
	part = PARTITIONS[parser.parse_args().num-1]
	outfile = lambda x: 'data/inputs/part/arrival_days.pkl' % x
	print('Partition: %s' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

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

	# convert to numpy arrays, save compressed
	path = 'data/inputs/%s/arrival_days.gz' % part
	f = h5py.File(path, 'w')
	for var in ['y', 'x_fixed', 'x_days']:
		array = globals()[var].to_numpy().astype('float32')
		f.create_dataset(var, data=array, dtype='float32')
	f.close()