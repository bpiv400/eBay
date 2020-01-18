import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, get_days_delay, get_norm
from processing.e_inputs.inputs_utils import load_file, get_x_thread, get_x_offer, init_x
from constants import SIM_CHUNKS, ENV_SIM_DIR, MONTH, IDX, SLR_PREFIX


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
	df['months_since_lstg'] = (df.clock - df.start_time) / MONTH
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


def construct_x(part, threads, offers):
	# master index
	idx = threads.index

	x = init_x(part, idx)

	# thread features
	x['lstg'] = pd.concat([x['lstg'], get_x_thread(threads, idx)], axis=1)

	# offer features
	x.update(get_x_offer(offers))

	return x


# loads data and calls helper functions to construct training inputs
def process_inputs(part, obs=None):
	
	if obs:
		

	# construct inputs from simulations
	lstg_start = load_file(part, 'lookup').start_time
	threads_sim = get_threads_sim(threads_obs.columns)
	offers_sim = get_offers_sim(offers_obs.columns)

	x = construct_x(part, )

	return x


def process_obs(part):
	# load inputs from data
		threads = load_file(part, 'x_thread') 
		offers = load_file(part, 'x_offer')


def main():
	# extract partition from command line
	part = input_partition()
	print('%s/threads' % part)

	# observed data
	x_obs = process_obs(part)

	# simulated data
	x_sim = process_sim(part, x_obs)

	# save data
	save_discrim_files(part, 'threads', x_obs, x_sim)


if __name__ == '__main__':
	main()