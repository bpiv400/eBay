import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome):
	# path name function
	getPath = lambda names: \
		'data/partitions/%s/%s.gz' % (part, '_'.join(names))

	# outcome
	y = load(getPath(['y_arrival', outcome]))

	# initialize fixed features with listing variables
	x_fixed = cat_x_lstg(part).reindex(index=y.index, level='thread')

	# add days since lstg start, holiday and day of week
	threads = load(getPath(['x', 'offer'])).xs(1, level='index')
	dow = [v for v in threads.columns if v.startswith('dow')]
	cols = ['days', 'holiday'] + dow
	x_fixed = x_fixed.join(threads[cols].rename(
		lambda x: 'focal_' + x, axis=1))

	return y, x_fixed


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int, help='Model ID.')
	num = parser.parse_args().num-1

	# partition and outcome
	k = len(OUTCOMES_ARRIVAL)
	part = PARTITIONS[num // k]
	outcome = OUTCOMES_ARRIVAL[num % k]
	outfile = lambda x: 'data/inputs/%s/arrival_%s.pkl' % (x, outcome)
	print('Model: arrival')
	print('Outcome: %s' % outcome)
	print('Partition: %s' % part)

	# input dataframes, output processed dataframes
	y, x_fixed = process_inputs(part, outcome)

	# save featnames and sizes once
	if part == 'train_models':
		# save featnames
		featnames = {'x_fixed': x_fixed.columns}
		pickle.dump(featnames, open(outfile('featnames'), 'wb'))

		# get data size parameters and save
		sizes = {'N': len(x_fixed.index), 'fixed': len(x_fixed.columns)}
		pickle.dump(sizes, open(outfile('sizes'), 'wb'))

	# convert to numpy
	data = {'y': y.to_numpy(),
			'x_fixed': x_fixed.to_numpy().astype('float32')}

	# out path
	path = lambda x:'data/inputs/%s/arrival_%s.%s' % (part, outcome, x)

	# save as hdf5
	f = h5py.File(path('hdf5'), 'w')
	for k, v in data.items():
		f.create_dataset(k, data=v, dtype=v.dtype)

	# save as gz
	dump(data, path('gz'))