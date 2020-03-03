import sys
import argparse
from compress_pickle import load
import numpy as np
from constants import INPUT_DIR, VALIDATION, CON_MODELS, MSG_MODELS, \
	FIRST_ARRIVAL_MODEL, BYR_HIST_MODEL, ARRIVAL_PREFIX
from featnames import CON, MSG


def log_likelihood(p):
	p = p[p > 0]
	return np.sum(p * np.log(p))


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--outcome', type=str)
	outcome = parser.parse_args().outcome
	assert outcome in [ARRIVAL_PREFIX, CON, MSG]

	# function
	if outcome == MSG:
		names = MSG_MODELS
	elif outcome == CON:
		names = CON_MODELS
	else:
		names = [FIRST_ARRIVAL_MODEL, BYR_HIST_MODEL]

	# load data
	
	for name in names:
		p = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, name))['p']
		print('{0}: {1:1.4f}'.format(name, log_likelihood(p)))


if __name__ == '__main__':
	main()
