import sys, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import get_x_offer, save_files
from processing.processing_consts import *


def get_tf(tf, start_time, role):
    # subset by role
    tf = tf[tf.index.isin(IDX[role], level='index')]
    # add period to tf_arrival
    tf = tf.reset_index('clock')
    tf = tf.join(start_time)
    tf['period'] = (tf.clock - tf.start_time) // INTERVAL[role]
    tf = tf.drop(['clock', 'start_time'], axis=1)
    # increment period by 1; time feats are up to t-1
    tf['period'] += 1
    # drop periods beyond censoring threshold
    tf = tf[tf.period < INTERVAL_COUNTS[role]]
    if role == 'byr':
        tf = tf[~tf.index.isin([7], level='index') | \
                (tf.period < INTERVAL_COUNTS['byr_7'])]
    # sum count features by period and return
    return tf.groupby(['lstg', 'thread', 'index', 'period']).sum()


def get_y(events, role):
	# construct delay
	clock = events.clock.unstack()
	delay = pd.DataFrame(index=clock.index)
	for i in range(2, 8):
		delay[i] = clock[i] - clock[i-1]
		delay.loc[delay[i] == 0, i] = np.nan  # remove auto responses
	delay = delay.rename_axis('index', axis=1).stack().astype('int64')
	# restrict to role indices
	s = delay[delay.index.isin(IDX[role], level='index')]
	c = events.censored.reindex(index=s.index)
	# expirations
	exp = s >= MAX_DELAY[role]
	if role == 'byr':
		exp.loc[exp.index.isin([7], level='index')] = s >= MAX_DELAY['slr']
	# interval of offer arrivals and censoring
	arrival = (s[~exp & ~c] / INTERVAL[role]).astype('uint16').rename('arrival')
	cens = (s[~exp & c] / INTERVAL[role]).astype('uint16').rename('cens')
	# initialize output dataframe with arrivals
	df = arrival.to_frame().assign(count=1).set_index(
		'arrival', append=True).squeeze().unstack(
		fill_value=0).reindex(index=s.index, fill_value=0)
	# vector of censoring thresholds
	v = (arrival+1).append(cens, verify_integrity=True).reindex(
		s.index, fill_value=INTERVAL_COUNTS[role])
	if role == 'byr':
		mask = v.index.isin([7], level='index') & (v > INTERVAL_COUNTS['byr_7'])
		v.loc[mask] = INTERVAL_COUNTS['byr_7']
	# replace censored observations with -1
	for i in range(INTERVAL_COUNTS[role]):
		df[i] -= (i >= v).astype('int8')
	# sort by turns and return
	return sort_by_turns(df)


# loads data and calls helper functions to construct train inputs
def process_inputs(part, role):
	# function to load file
	load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

	# load dataframes
	lstg_start = load_file('lookup').start_time
	events = load(CLEAN_DIR + 'offers.pkl')[['clock', 'censored']].reindex(
		index=lstg_start.index, level='lstg')

	# outcome
	y = get_y(events, role)
	idx = y.index

	# dictionary of input features
	x = get_x_offer(load_file, idx, outcome='delay', role=role)

	# second since START for each observation
	seconds = events.clock.groupby(
		['lstg', 'thread']).shift().reindex(index=idx).astype('int64')

	# normalized periods remaining at start of delay period
	remaining = MAX_DAYS * 24 * 3600 - (seconds - lstg_start)
	remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
		MAX_DELAY['slr']
	remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
		MAX_DELAY['byr']
	remaining = np.minimum(remaining, 1)

	# time features
	tf = get_tf(load_file('tf_delay'), idx_clock, role)

	return {'y': y, 'periods': periods, 'x': x,
			'seconds': seconds, 'remaining': remaining, 'tf': tf}


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
