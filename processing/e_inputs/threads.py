import sys, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, get_days_delay, get_norm
from processing.e_inputs.inputs_utils import load_file, get_x_thread, get_x_offer, init_x
from utils import is_split
from constants import SIM_CHUNKS, ENV_SIM_DIR, MONTH, IDX, SLR_PREFIX
from featnames import CON, DAYS, DELAY, EXP, AUTO, REJECT, MONTHS_SINCE_LSTG


def process_offers_sim(df, offer_cols):
	# do stuff
	
	
	df[DAYS], df[DELAY] = get_days_delay(df.clock.unstack())

	# concession as a decimal
	df.loc[:, CON] /= 100

	# indicator for split
    df[SPLIT] = is_split(df[CON])

	# total concession
    df[NORM] = get_norm(df[CON])
    
    # reject auto and exp are last
    df[REJECT] = df[CON] == 0
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df[EXP] = (df[DELAY] == 1) | df[CENSORED]

	# reorder columns to match observed
	df = df[offer_cols]

	return df

def process_threads_sim(df, thread_cols, lstg_start):
	# convert clock to months_since_lstg
	df = df.join(lstg_start)
	df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
	df = df.drop(['clock', 'start_time'], axis=1)
	# reorder columns to match observed
	df = df[thread_cols]
	return df


def concat_sim_chunks(part):
	'''
	Loops over simulations, concatenates dataframes.
	:param part: string name of partition.
	:return: concatentated and sorted threads and offers dataframes.
	'''
	threads_sim, offers_sim = [], []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
		threads_sim.append(sim['threads'])
		offers_sim.append(sim['offers'])
	threads_sim = pd.concat(threads_sim, axis=0).sort_index()
	offers_sim = pd.concat(offers_sim, axis=0).sort_index()
	return threads_sim, offers_sim


def process_sim(part, thread_cols, offer_cols):
	lstg_start = load_file(part, 'lookup').start_time

	# construct inputs from simulations
	threads_sim, offers_sim = concat_sim_chunks(part)

	threads_sim = process_threads_sim(threads_sim, thread_cols, lstg_start)
	offers_sim = process_offers_sim(offers_sim, offer_cols)

	x = construct_x(part, )

	return x


def construct_x(part, threads, offers):
	# master index
	idx = threads.index

	# initialize input dictionary with lstg features
	x = init_x(part, idx)

	# add thread features to x['lstg']
	x['lstg'] = pd.concat([x['lstg'], get_x_thread(threads, idx)], axis=1)

	# offer features
	x.update(get_x_offer(offers, idx))
	
	return x


def process_obs(part):
	# load inputs from data
	threads_obs = load_file(part, 'x_thread') 
	offers_obs = load_file(part, 'x_offer')
	# dictionary of input features
	x_obs = construct_x(part, threads_obs, offers_obs)
	# return input features and dataframe columns
	return x_obs, threads_obs.columns, offers_obs.columns


def main():
	# extract partition from command line
	part = input_partition()
	print('%s/threads' % part)

	# observed data
	x_obs, thread_cols, offer_cols = process_obs(part)

	# simulated data
	x_sim = process_sim(part, thread_cols, offer_cols)

	# save data
	save_discrim_files(part, 'threads', x_obs, x_sim)


if __name__ == '__main__':
	main()