import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_sim_counts(lstg_start):
	# concatenate thread start times
	l = []
	for i in range(1, SIM_CHUNKS+1):
		# get thread start times from first simulation
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
		l.append(sim['threads'].clock.xs(0, level='sim'))
	thread_start = pd.concat(l, axis=0)

	# convert to period in range(INTERVAL['arrival'])
	diff = (thread_start - lstg_start.reindex(
		index=thread_start.index, level='lstg'))
	period = (diff // INTERVAL['arrival']).rename('period')

	# count arrivals and fill in zeros
	counts = period.to_frame().assign(count=1).set_index(
		'period', append=True).groupby(['lstg', 'period']).sum().squeeze()
	counts = counts.unstack(fill_value=0).rename_axis(
		'', axis=1).reindex(index=lstg_start.index, fill_value=0)

	return counts


def process_inputs(part):
	# listing start time
	lstg_start = load('{}{}/lookup.gz'.format(PARTS_DIR, part)).start_time
	idx = lstg_start.index

	# load dataframe of real counts
	obs_counts = load('{}{}/y_arrival.gz'.format(PARTS_DIR, part)).reindex(
		index=idx).clip(lower=0)	# swap -1s for 0s

	# construct corresponding dataframe of simulated counts
	sim_counts = get_sim_counts(lstg_start)

	# other inputs
	x_other = pd.concat([obs_counts, sim_counts], axis=0)

	# y=True indicates observed
	y_obs = pd.Series(1, index=idx, dtype='int8')
	y_sim = pd.Series(0, index=idx, dtype='int8')
	y = pd.concat([y_obs, y_sim], axis=0)

	# create listing features
	x = init_x(part)

	# ensure indices are correct
	assert np.all(x_other.index == y.index)
	assert all([np.all(v.index == idx) for v in x.values()])

	# index of x
	s = pd.Series(range(len(idx)), index=idx)
	x_idx = pd.concat([s, s], axis=0)

	# combine into single dictionary
	return {'y': y.astype('int8', inplace=True), 'x': x,
			'x_other': x_other, 'x_idx': x_idx}


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	args = parser.parse_args()
	part = args.part

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save featnames and sizes
	if part == 'train_rl':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/listings.pkl' % PREFIX, 'wb'))

		pickle.dump(get_sizes(d), 
			open('%s/inputs/sizes/listings.pkl' % PREFIX, 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(d, '%s/inputs/%s/listings.gz' % (PREFIX, part))
