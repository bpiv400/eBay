import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_arrival_counts(lstg):
	return None


def get_sim_counts(df):
	counts_sim = pd.DataFrame(0, index=df.index, 
		columns=df.columns, dtype=df.dtype)
	for lstg in df.index:
		counts_sim.loc[lstg, :] = get_arrival_counts(lstg)
	return counts_sim


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	args = parser.parse_args()
	part = args.part

	# listing features
	x = init_x(part)
	N = len(x['lstg'].index)

	# load dataframe of real counts, swap -1s for 0s
	counts_obs = load(PARTS_DIR + '%s/y_arrival.gz' % part).reindex(
		index=x['lstg'].index).clip(lower=0)

	# construct corresponding dataframe of simulated counts
	counts_sim = get_sim_counts(counts_obs)

	# repeat entries in x
	x = {k: pd.concat([v, v], axis=0) for k, v in x.items()}

	# add in arrivals
	x['arrivals'] = pd.concat([counts_obs, counts_sim], axis=0)

	# y=True indicates observed
	y = pd.Series(True, index=x['arrivals'].index, dtype=bool)
	y.iloc[N:] = False

	# combine into single dictionary
	d = {'y': y.astype('int8', inplace=True), 'x': x}

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/listings.pkl' % PREFIX, 'wb'))

		pickle.dump(get_sizes(d, model), 
			open('%s/inputs/sizes/listings.pkl' % PREFIX, 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(d, '%s/inputs/%s/listings.gz' % (PREFIX, part))

	# save small dataset
	if part == 'train_models':
		dump(create_small(d), '%s/inputs/small/listings.gz' % PREFIX)
