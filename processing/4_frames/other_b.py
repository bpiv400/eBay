from compress_pickle import load, dump
import pandas as pd
import argparse, sys

sys.path.append('repo/')
from constants import *


if __name__ == "__main__":
	# partition number from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num

	part = PARTITIONS[num-1]

	path = 'data/partitions/%s/x_offer.gz' % part

	x_offer = load(path)

	x_offer.loc[x_offer.norm.isna(), 'norm'] = 0

	dump(x_offer, path)