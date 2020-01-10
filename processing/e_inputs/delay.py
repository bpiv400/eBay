import sys, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import get_x_thread, get_x_offer, get_idx_x, \
	save_files, load_file, get_tf, periods_to_string, get_first_index
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS, MAX_DELAY
from constants import DAY, MONTH, IDX


def get_delay(x_offer):
	# convert to seconds
	delay = np.round(x_offer.days * DAY).astype('int64')

	# error checking
	assert delay.max() <= MAX_DELAY[role]
	if role == 'byr':
		assert delay.xs(7, level='index').max() <= MAX_DELAY['slr']

	return delay


def get_periods(delay, role):
	# convert to interval
	periods = (delay // INTERVAL[role]).rename('periods')

	# error checking
	assert periods.max() <= INTERVAL_COUNTS[role]

	# minimum number of periods is 1
	periods.loc[periods < INTERVAL_COUNTS[role]] += 1

	# sort and return
	return periods


# loads data and calls helper functions to construct train inputs
def process_inputs(part, role):
	# load features from offer dataframe and restrict observations
	x_offer = load_file(part, 'x_offer')[['days', 'exp']]
	x_offer = x_offer.loc[x_offer.days > 0]
	x_offer = x_offer.loc[x_offer.index.isin(IDX[role], level='index')]

	# delay in seconds
	delay = get_delay(x_offer)

	# number of periods
	periods = get_periods(delay, role)
	idx = periods.index

	# outcome
	y = ~x_offer.exp

	# thread features
	x_thread = get_x_thread(part, idx)

	# offer features
	x_offer = get_x_offer(part, idx, outcome='delay', role=role)

	# index of listing features
	idx_x = get_idx_x(part, idx)

	# second since START at beginning of delay period
	clock = load_file(part, 'clock')
	delay_start = clock.groupby(['lstg', 'thread']).shift().reindex(
		index=idx).astype('int64').rename('delay_start')

	# normalized periods remaining at start of delay period
	lstg_start = load_file(part, 'lookup').start_time
	diff = delay_start - lstg_start.reindex(index=idx, level='lstg')
	remaining = MONTH - diff
	remaining.loc[idx.isin([2, 4, 6, 7], level='index')] /= MAX_DELAY['slr']
	remaining.loc[idx.isin([3, 5], level='index')] /= MAX_DELAY['byr']
	remaining = np.minimum(remaining, 1)

	# time features
	tf = load_file(part, 'tf_delay')
	tf = tf[tf.index.isin(IDX[role], level='index')]
	tf_delay = get_tf(tf, delay_start, periods, role)

	# periods in which non-zero time features are observed
	tf_periods = periods_to_string(tf_delay.index, idx)

	# first index of time features for each lstg
	idx_tf = get_first_index(tf_delay.index, idx)

	# error checking
	assert all(tf_periods[idx_tf == -1] == ''.encode('ascii'))
	assert all(idx_tf[tf_periods == ''.encode('ascii')] == -1)

	# dictionary of input components
	return {'periods': periods, 
			'y': y, 
			'x_thread': x_thread,
			'x_offer': x_offer,
			'idx_x': idx_x,
			'seconds': delay_start, 
			'remaining': remaining, 
			'tf': tf_delay,
            'tf_periods': tf_periods,
            'idx_tf': idx_tf}


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--role', type=str)
	args = parser.parse_args()
	part, role = args.part, args.role
	name = 'delay_%s' % role
	print('%s/%s' % (part, name))

	# input dataframes, output processed dataframes
	d = process_inputs(part, role)

	# save various output files
	save_files(d, part, name)
