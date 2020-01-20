import sys, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd

from processing.e_inputs.inputs_utils import load_file, get_x_thread, \
	get_x_offer, init_x, save_files
from utils import get_remaining

from processing.processing_consts import MAX_DELAY, INTERVAL, CLEAN_DIR
from constants import IDX, DAY, BYR_PREFIX, SLR_PREFIX, ARRIVAL_PREFIX
from featnames import CON, MSG, AUTO, EXP, REJECT, DAYS, DELAY, INT_REMAINING


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


def calculate_remaining(part, idx, role):
	# load timestamps
	lstg_start = load_file(part, 'lookup').start_time.reindex(
			index=idx, level='lstg')
	delay_start = load_file(part, 'clock').groupby(
		['lstg', 'thread']).shift().reindex(index=idx).astype('int64')

	# maximal delay
	max_delay = pd.Series(MAX_DELAY[role], index=idx)
	max_delay.loc[max_delay.index.isin([7], level='index')] = \
		MAX_DELAY[BYR_PREFIX + '_7']

	# remaining calculation
	remaining = get_remaining(lstg_start, delay_start, max_delay)

	# error checking
	assert np.all(remaining > 0) and np.all(remaining <= 1)

	return remaining


def check_zero(offer, cols):
	for c in cols:
		assert offer[c].max() == 0
		assert offer[c].min() == 0


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

	# listing features
	x = init_x(part, idx)

	# thread features
	x_thread = get_x_thread(threads, idx)

	if outcome == 'delay':	# add time remaining to x_thread
		x_thread[INT_REMAINING] = calculate_remaining(part, idx, role)

	x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1) 

	# offer features
	x.update(get_x_offer(offers, idx, outcome=outcome, role=role))

	# error checking
	for i in range(1, max(IDX[role])+1):
		# offer features at turn i
		offer = x['offer' + str(i)]
		# error checking
		if i == 1:
			check_zero(offer, [DAYS, DELAY])
		if i % 2 == 1:
			check_zero(offer, [AUTO, EXP, REJECT])
		if i == 7:
			check_zero(offer, [MSG])

	return {'y': y, 'x': x}


def main():
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


if __name__ == '__main__':
	main()