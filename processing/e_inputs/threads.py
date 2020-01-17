import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, load_file, save_files, \
	get_x_thread, get_x_offer, get_idx_x, get_days_delay, get_norm
from constants import SIM_CHUNKS, ENV_SIM_DIR, MAX_DAYS, DAY, IDX, SLR_PREFIX


def concat_sim_chunks(name):
	'''
	Loops over simulations, concatenates single dataframe.
	:param name: either 'threads' or 'offers'
	:return: sorted dataframe with concatenated data.
	'''
	assert name in ['threads', 'offers']
	df = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
		df.append(sim[name])
	df = pd.concat(df, axis=0).sort_index()
	return df


def get_threads_sim(cols, lstg_start):
	# concatenate chunks
	df = concat_sim_chunks('threads')

	# convert clock to months_since_lstg
	df = df.join(lstg_start)
	df['months_since_lstg'] = (df.clock - df.start_time) / (MAX_DAYS * DAY)
	df = df.drop(['clock', 'start_time'], axis=1)

	# reorder columns to match observed
	df = df[cols]

	return df


def get_offers_sim(cols):
	# concatenate chunks
	df = concat_sim_chunks('offers')

	# do stuff
	
	
	df['days'], df['delay'] = get_days_delay(df.clock.unstack())

	# concession as a decimal
	df.loc[:, 'con'] /= 100

	# total concession
    df['norm'] = get_norm(df.con)

    # indicator for split
    df['split'] = (df.con >= 0.49) & (df.con <= 0.51)

    # message indicator
    df = df.rename({'message': 'msg'}, axis=1)

    # reject auto and exp are last
    df['reject'] = df.con == 0
    df['auto'] = (df.delay == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df['exp'] = (df.delay == 1) | events.censored


	# reorder columns to match observed
	df = df[cols]

	return df


def construct_components(threads, offers):
	idx = threads.idx

	# thread features
	x_thread = get_x_thread(threads, idx)

	# offer features
	x_offer = get_x_offer(offers, idx, outcome='threads')

	# index of listing features
	idx_x = get_idx_x(part, idx)

	return x_thread, x_offer, idx_x


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome, role):
	# construct inputs from data
	threads_obs = load_file(part, 'x_thread') 
	offers_obs = load_file(part, 'x_offer')
	x_thread_obs, x_offer_obs, idx_x_obs = construct_components(threads, offers)

	# construct inputs from simulations
	lstg_start = load_file(part, 'lookup').start_time
	threads_sim = get_threads_sim(threads_obs.columns)
	offers_sim = get_offers_sim(offers_obs.columns)

	return {'y': y, 
			'x_thread': x_thread, 
			'x_offer': x_offer, 
			'idx_x': idx_x}


if __name__ == '__main__':
	# extract partition from command line
	part = input_partition()

	# input dataframes, output processed dataframes
	d = process_inputs(part)


