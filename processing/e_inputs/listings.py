from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
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
	# function to load file
	load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

	# listing start time
	lstg_start = load_file('lookup').start_time
	idx = lstg_start.index

	# load dataframe of real counts and swap -1s for 0s
	obs_counts = load_file('y_arrival').reindex(index=idx).clip(lower=0)

	# construct corresponding dataframe of simulated counts
	sim_counts = get_sim_counts(lstg_start)

	# other inputs
	x_arrival = pd.concat([obs_counts, sim_counts], axis=0, copy=False)

	# y=True indicates observed
	y_obs = pd.Series(1, index=idx, dtype='int8')
	y_sim = pd.Series(0, index=idx, dtype='int8')
	y = pd.concat([y_obs, y_sim], axis=0)

	# lstg input features, stacked
	x = load_file('x_lstg')
	x = {k: pd.concat([v, v], axis=0) for k, v in x.items()}

	# add in arrival features
	x['arrival'] = pd.concat([obs_counts, sim_counts], axis=0)

	# ensure indices are correct
	assert all([np.all(v.index == y.index) for v in x.values()])

	# combine into single dictionary
	return {'y': y.astype('int8', inplace=True), 
			'x': {k: v.astype('float32', copy=False) for k, v in x.items()}}


if __name__ == '__main__':
	# extract partition from command line
	part = input_partition()

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save sizes
	if part == 'train_rl':
		dump(get_sizes(d), '{}/inputs/sizes/listings.pkl'.format(PREFIX))

	# convert to dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save
	dump(d, '{}/inputs/{}/listings.gz'.format(PREFIX, part))

	# small arrays
	if part == 'train_rl':
        dump(create_small(d), '{}/inputs/small/listings.gz'.format(PREFIX))