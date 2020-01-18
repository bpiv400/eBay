import sys, os, argparse
import pandas as pd, numpy as np
from compress_pickle import load, dump
from processing.processing_utils import save_files
from constants import INPUT_DIR


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--name', type=str)
	args = parser.parse_args()
	part, name = args.part, args.name
	print('%s/%s' % (part, name))

	# path
	path = INPUT_DIR + '{}/{}.gz'.format(part, name)

	# load
	d = load(path)

	# convert to numpy
	d['periods'] = d['periods'].to_numpy()

	# save
	dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))