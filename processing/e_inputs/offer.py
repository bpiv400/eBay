import sys, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import get_x_thread, get_x_offer, \
	get_idx_x, save_files, load_file
from processing.processing_consts import MAX_DELAY, INTERVAL, CLEAN_DIR
from constants import IDX, DAY, BYR_PREFIX, SLR_PREFIX, ARRIVAL_PREFIX
from featnames import CON, MSG, AUTO, EXP, DAYS, INT_REMAINING


def get_y_con(df):
	# drop zero delay and expired offers
	mask = ~df[AUTO] & ~df[EXP]

	# concession is an int from 0 to 100
	return (df.loc[mask, CON] * 100).astype('int8')


def get_y_msg(df):
	# drop accepts and rejects
	mask = (df[CON] > 0) & (df[CON] < 1)
	return df.loc[mask, MSG]


def get_y_delay(df, role):
	# convert to seconds
	delay = np.round(df[DAYS] * DAY).astype('int64')

	# drop zero delays
	delay = delay[delay > 0]

	# error checking
	assert delay.max() <= MAX_DELAY[role]
	if role == BYR_PREFIX:
		assert delay.xs(7, level='index').max() <= MAX_DELAY[SLR_PREFIX]

	# replace censored delays with negative seconds from end
	delay.loc[df[EXP]] -= MAX_DELAY[role] + 1

	# convert to periods
	delay //= INTERVAL[role]

	return delay


# loads data and calls helper functions to construct train inputs
def process_inputs(part, outcome, role):
	# load dataframes
	offers = load_file(part, 'x_offer')
	threads = load_file(part, 'x_thread')

	# outcome and master index
	df = offers[offers.index.isin(IDX[role], level='index')]
	if outcome == 'con':
		y = get_y_con(df)
	elif outcome == 'msg':
		y = get_y_msg(df)
	elif outcome == 'delay':
		y = get_y_delay(df, role)
	idx = y.index

	# thread features
	x_thread = get_x_thread(threads, idx)

	# add time remaining to x_thread
	if outcome == 'delay':
		delay_start = clock.groupby(['lstg', 'thread']).shift().reindex(
			index=idx).astype('int64')
		lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
			index=idx, level='lstg')
		x_thread[INT_REMAINING] = np.minimum(1,
			(lstg_end - delay_start) / MAX_DELAY[role])

	# offer features
	x_offer = get_x_offer(offers, idx, outcome=outcome, role=role)

	# index of listing features
	idx_x = get_idx_x(part, idx)

	return {'y': y, 
			'x_thread': x_thread, 
			'x_offer': x_offer, 
			'idx_x': idx_x}


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--outcome', type=str)
	parser.add_argument('--role', type=str)
	args = parser.parse_args()
	part, outcome, role = args.part, args.outcome, args.role
	name = '%s_%s' % (outcome, role)
	print('%s/%s' % (part, name))

	# input dataframes, output processed dataframes
	d = process_inputs(part, outcome, role)

	# save various output files
	save_files(d, part, name)
