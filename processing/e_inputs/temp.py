import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *

MODELS = ['hist', 'con_slr', 'con_byr', 'msg_slr', 'msg_byr']

if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1
	name = MODELS[num]

	d = load('%s/inputs/train_models/%s.gz' % (PREFIX, name))

	small = create_small(d)

	dump(small, '%s/inputs/small/%s.gz' % (PREFIX, name))