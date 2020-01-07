import sys, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import get_x_offer, save_files, load_file
from constants import IDX


def get_y_con(x_offer):
	# drop zero delay and expired offers
	mask = ~x_offer.auto & ~x_offer.exp
	# concession is an int from 0 to 100
	return (x_offer.loc[mask, 'con'] * 100).astype('int8')


def get_y_msg(x_offer):
	# drop accepts and rejects
	mask = (x_offer.con > 0) & (x_offer.con < 1)
	return x_offer.loc[mask, 'msg']


# loads data and calls helper functions to construct train inputs
def process_inputs(part, outcome, role):
	# load offer data
	x_offer = load_file(part, 'x_offer')

	# subset by role indices
	x_offer = x_offer[x_offer.index.isin(IDX[role], level='index')]

	if outcome == 'con':
		y = get_y_con(x_offer)
	elif outcome == 'msg':
		y = get_y_msg(x_offer)
	idx = y.index

	# dictionary of input features
	x = get_x_offer(part, idx, outcome=outcome, role=role)

	return {'y': y, 'x': x}


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
