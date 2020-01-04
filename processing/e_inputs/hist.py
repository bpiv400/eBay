import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import input_partition, save_files, load_file
from processing.processing_consts import *


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	# thread features
	x_offer = load_file(part, 'x_offer').xs(1, level='index')
	x_offer = x_offer.drop(['days', 'delay', 'con', 'norm', 'split', \
		'msg', 'reject', 'auto', 'exp'], axis=1)
	x_offer = x_offer.rename(lambda x: x + '_1', axis=1)
	x_thread = load_file(part, 'x_thread').join(x_offer)

	# outcome
	y = x_thread['byr_hist']
	idx = y.index
	x_thread.drop('byr_hist', axis=1, inplace=True)

	# initialize input features
	x = load_file(part, 'x_lstg')
	x = {k: v.reindex(index=idx, level='lstg') for k, v in x.items()}

	# add thread variables to x['lstg']
	x['lstg'] = x['lstg'].join(x_thread)

	return {'y': y, 'x': x}


if __name__ == '__main__':
	# partition name from command line
	part = input_partition()
	print('%s/hist' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, 'hist')
