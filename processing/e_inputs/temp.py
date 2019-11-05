import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *

MODELS = ['arrival', 'hist', 'con_slr', 'con_byr', 'msg_slr', 'msg_byr']


if __name__ == '__main__':
	# extract model from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and model
	part = PARTITIONS[num // len(MODELS)]
	model = MODELS[num % len(MODELS)]
	filename = '%s/inputs/%s/%s.gz' % (PREFIX, part, model)
	print(filename)

	# load dictionary
	d = load(filename)

	# add groups
	if 'turns' in d:
		d['groups'] = [np.nonzero(d['turns'] == n)[0] \
			for n in np.unique(d['turns'])]
	else:
		d['groups'] = [np.array(range(np.shape(d['x_fixed'])[0]))]

	# save dictionary
	dump(d, filename)