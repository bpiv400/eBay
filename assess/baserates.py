import sys
import argparse
from compress_pickle import load, dump
import numpy as np
from processing.processing_consts import NUM_OUT
from constants import INPUT_DIR, VALIDATION, MODELS, PLOT_DATA_DIR


def get_baserate(y, intervals):
	p = np.zeros(intervals, dtype='float64')
	for i in range(intervals):
		p[i] = (y == i).mean()
	p = p[p > 0]
	return np.sum(p * np.log(p))


def main():
	# initialize output dataframe
	lnL0, lnL_bar = dict(), dict()

	# calculate initialization value and baserate for each model
	for m in MODELS:
		if 'delay' not in m and m != 'next_arrival':
			print(m)

			# number of intervals
			intervals = NUM_OUT[m]
			if intervals == 1:
				intervals += 1

			# initialization value
			lnL0[m] = np.log(np.ones(intervals, dtype='float64') / intervals)

			# load data
			y = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, m))['y']

			# baserates
			lnL_bar[m] = get_baserate(y, intervals)

	# save output
	dump(lnL0, PLOT_DATA_DIR + '{}/{}.pkl'.format('lnL', 'lnL0'))
	dump(lnL_bar, PLOT_DATA_DIR + '{}/{}.pkl'.format('lnL', 'lnL_bar'))


if __name__ == '__main__':
	main()
