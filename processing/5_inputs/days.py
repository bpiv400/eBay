import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
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

	# index of x_fixed for each y
	lookup = np.array(range(len(x_fixed.index)))
	counts = y.groupby('lstg').count().values
	idx_fixed = np.repeat(lookup, counts)

	# day features
	N = pd.to_timedelta(pd.to_datetime(END) - pd.to_datetime(START)).days
	clock = pd.to_datetime(range(N+30+1), unit='D', origin=START)
	x_days = pd.Series(clock, name='clock')
	x_days = extract_day_feats(x_days).join(x_days).set_index('clock')

	# index of x_days for each y
	period = y.reset_index('period')['period']
	idx_days = (period + x_fixed.start_date).values

	return y, x_fixed, idx_fixed, x_days, idx_days


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and outcome
	part = PARTITIONS[num]
	outfile = 'data/inputs/%s/arrival_days.pkl' % part
	print('Model: arrival')
	print('Outcome: days')
	print('Partition: %s' % part)

	# input dataframes, output processed dataframes
	y, x_fixed, idx_fixed, x_days, idx_days = process_inputs(part)

	# save featnames and sizes once
	if part == 'train_models':
		# save featnames
		featnames = {'x_fixed': list(x_fixed.columns) + \
			list(x_days.rename(lambda x: x + '_focal', axis=1).columns)}
		pickle.dump(featnames, open(outfile('featnames'), 'wb'))

		# get data size parameters and save
		sizes = {'N': len(y.index), 
				 'fixed': len(x_fixed.columns) + len(x_days.columns),
				 'out': 2}
		pickle.dump(sizes, open(outfile('sizes'), 'wb'))

	# convert to numpy arrays, save in hdf5
	path = 'data/inputs/%s/arrival_days.hdf5' % part
	f = h5py.File(path, 'w')

	f.create_dataset('y', data=y.to_numpy(), dtype='uint8')

	for var in ['x_fixed', 'x_days']:
		array = globals()[var].to_numpy().astype('float32')
		f.create_dataset(var, data=array, dtype='float32')

	f.create_dataset('idx_fixed', data=idx_fixed.astype('uint32'), 
		dtype='uint32')
	f.create_dataset('idx_days', data=idx_days.astype('uint16'), 
		dtype='uint16')

	f.close()