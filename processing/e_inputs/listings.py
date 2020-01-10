from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import get_x_thread, get_x_offer, \
	get_idx_x, save_files, load_file, get_arrival, input_partition, \
	reshape_indices
from processing.processing_consts import *


def get_counts_sim(lstg_start):
	# concatenate thread start times
	l = []
	for i in range(1, SIM_CHUNKS+1):
		# get thread start times from first simulation
		try:
			sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
			l.append(sim['threads'].clock)
		except:
			continue
	thread_start = pd.concat(l, axis=0)

	# convert to period in range(INTERVAL['arrival'])
	diff = (thread_start - lstg_start.reindex(
		index=thread_start.index, level='lstg'))
	period = (diff // INTERVAL['arrival']).rename('period')

	# count arrivals and fill in zeros
	counts_sim = period.to_frame().assign(count=1).groupby(
		['lstg', 'period']).sum().squeeze()

	return counts_sim


def add_sim_to_index(df, isSim):
	return df.reset_index().assign(sim=isSim).set_index(
		['lstg', 'sim', 'period']).squeeze()


def process_inputs(part):
	# start times
	lstg_start = load_file(part, 'lookup').start_time
	thread_start = load_file(part, 'clock').xs(1, level='index')

	# construct complete index
	idx = pd.MultiIndex.from_product(
			[lstg_start.index, [True, False]], 
			names=['lstg', 'sim'])

	# observed arrival counts
	counts_obs = get_arrival(lstg_start, thread_start)

	# construct corresponding dataframe of simulated counts
	counts_sim = get_counts_sim(lstg_start)

	# add 'sim' to counts index
	counts_obs = add_sim_to_index(counts_obs, False)
	counts_sim = add_sim_to_index(counts_sim, True)

	# other inputs
	arrival = pd.concat([counts_obs, counts_sim], 
		axis=0, copy=False).sort_index()

	# periods and arrival indices with index idx
	arrival_periods, idx_arrival = reshape_indices(arrival.index, idx)

	# y=True indicates simulated
	y = pd.Series(idx.get_level_values(level='sim'), 
		index=idx, name='isSim').astype('bool')

	# index of listing features
	idx_x = get_idx_x(part, idx)

	# combine into single dictionary
	return {'y': y, 
			'arrival': arrival,
			'arrival_periods': arrival_periods,
			'idx_arrival': idx_arrival,
			'idx_x': idx_x}


if __name__ == '__main__':
	# extract partition from command line
	part = input_partition()
	print('%s/listings' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, 'listings')
