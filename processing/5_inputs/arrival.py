import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *


def add_thread_feats(outcome, x_fixed, x_thread):
    # return or add features
    x_fixed = x_fixed.join(x_thread['byr_us'])
    if outcome == 'hist':
        return x_fixed
    x_fixed = x_fixed.join(x_thread['byr_hist'])
    if outcome in ['bin', 'sec']:
        return x_fixed


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome):
	# path name function
	getPath = lambda names: \
		'data/partitions/%s/%s.gz' % (part, '_'.join(names))

	# outcome
	y = load(getPath(['y_arrival', outcome]))

	# initialize fixed features with thread ids and listing variables
	threads = load(getPath(['x', 'offer'])).xs(
		1, level='index').reindex(index=y.index)
	x_fixed = pd.DataFrame(index=threads.index).join(cat_x_lstg(part))

	# add days since lstg start, holiday and day of week
	dow = [v for v in threads.columns if v.startswith('dow')]
	cols = ['days', 'holiday'] + dow
	x_fixed = x_fixed.join(threads[cols].rename(
		lambda x: 'focal_' + x, axis=1))

	# add byr_us
	if outcome != 'loc':
		x_thread = load(getPath(['x', 'thread']))
		x_fixed = add_thread_feats(outcome, x_fixed, x_thread)

	return y, x_fixed


def get_sizes(outcome, x_fixed):
    sizes = {}

    # number of observations
    sizes['N'] = len(x_fixed.index)

    # fixed inputs
    sizes['fixed'] = len(x_fixed.columns)

    # output parameters
    if outcome == 'sec':
        sizes['out'] = 3
    elif outcome == 'hist':
        sizes['out'] = 2
    else:
        sizes['out'] = 1

    return sizes


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
		sizes = get_sizes(outcome, x_fixed)
		pickle.dump(sizes, open(outfile('sizes'), 'wb'))

	# convert to numpy arrays, save in hdf5
	path = 'data/inputs/%s/arrival_%s.hdf5' % (part, outcome)
	f = h5py.File(path, 'w')
	for var in ['y', 'x_fixed']:
		array = globals()[var].to_numpy().astype('float32')
		f.create_dataset(var, data=array, dtype='float32')
	f.close()