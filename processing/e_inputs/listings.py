from compress_pickle import load
import numpy as np, pandas as pd
from processing.processing_utils import input_partition
from processing.e_inputs.inputs_utils import load_file, \
	process_arrival_inputs, save_discrim_files
from processing.processing_consts import INTERVAL_COUNTS
from constants import SIM_CHUNKS, ENV_SIM_DIR, MONTH, ARRIVAL_PREFIX


def process_lstg_end(lstg_start, lstg_end):
	# remove thread and index from lstg_end index
	lstg_end = lstg_end.reset_index(['thread', 'index'], drop=True)
	assert not lstg_end.index.duplicated().max()
	# fill in missing lstg end times with expirations
	lstg_end = lstg_end.reindex(index=lstg_start.index, fill_value=-1)
	lstg_end.loc[lstg_end == -1] = lstg_start + MONTH - 1
	return lstg_end


def get_sim_times(part, lstg_start):
	# collect times from simulation files
	lstg_end, thread_start = [], []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
		offers, threads = [sim[k] for k in ['offers', 'threads']]
		lstg_end.append(offers.loc[offers.con == 100, 'clock'])
		thread_start.append(threads.clock)
	# concatenate into single series
	lstg_end = pd.concat(lstg_end, axis=0).sort_index()
	thread_start = pd.concat(thread_start, axis=0).sort_index()
	# shorten index and fill-in expirations
	lstg_end = process_lstg_end(lstg_start, lstg_end)
	return lstg_end, thread_start


def get_obs_times(part, lstg_start):
	# offer timestamps
	clock = load_file(part, 'clock')
	# listing end time
	sale = load_file(part, 'x_offer').con == 1
	lstg_end = clock[sale]
	lstg_end = process_lstg_end(lstg_start, lstg_end)
	# thread start time
	thread_start = clock.xs(1, level='index')
	return lstg_end, thread_start


def process_inputs(part, obs=None):
	# listing start time
	lstg_start = load_file(part, 'lookup').start_time
	# listing end time and thread start time
	if obs:
		lstg_end, thread_start = get_obs_times(part, lstg_start)
	else:
		lstg_end, thread_start = get_sim_times(part, lstg_start)
	# dictionary of y and x
	d = process_arrival_inputs(part, lstg_start, lstg_end, thread_start)
	# restrict to first interarrival period
	d['y'] = d['y'].xs(1, level='thread')
	d['x'] = {k: v.xs(1, level='thread') for k, v in d['x'].items()}
	# split into arrival indicator and period number
	arrival = pd.Series(d['y'] >= 0, 
		index=d['y'].index, name='arrival', dtype='float32')
	period = pd.Series(d['y'], 
		index=d['y'].index, name='period', dtype='float32')
	period[period < 0] += INTERVAL_COUNTS[ARRIVAL_PREFIX] + 1
	period /= INTERVAL_COUNTS[ARRIVAL_PREFIX]
	# put y in x['lstg']
	d['x']['lstg'] = pd.concat([d['x']['lstg'], arrival, period], axis=1)
	return d['x']


def main():
	# extract partition from command line
	part = input_partition()
	print('%s/listings' % part)

	# observed data
	x_obs = process_inputs(part, obs=True)

	# simulated data
	x_sim = process_inputs(part, obs=False)

	# save data
	save_discrim_files(part, 'listings', x_obs, x_sim)


if __name__ == '__main__':
	main()