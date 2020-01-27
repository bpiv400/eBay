from compress_pickle import load
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, load_file, process_arrival_inputs
from processing.f_discrim.discrim_utils import save_discrim_files, get_sim_times
from processing.processing_consts import INTERVAL_COUNTS
from constants import ARRIVAL_PREFIX


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
	arrival = pd.Series(d['y'] < INTERVAL_COUNTS[ARRIVAL_PREFIX], 
		index=d['y'].index, name='arrival', dtype='float32')
	period = pd.Series(d['y'], 
		index=d['y'].index, name='period', dtype='float32')
	period /= INTERVAL_COUNTS[ARRIVAL_PREFIX]	# redefine on [0,1]
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