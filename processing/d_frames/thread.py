import sys, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


if __name__ == "__main__":
	# partition number from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num-1

	# partition
	part = PARTITIONS[num]
	idx, path = get_partition(part)

	# load data
	lookup = load(PARTS_DIR + '%s/lookup.gz' % part)
	threads = load(CLEAN_DIR + 'threads.pkl').reindex(
		index=idx, level='lstg')

	# initialize x_thread with calendar features
	clock = pd.to_datetime(threads['start_time'], unit='s', origin=START)
	x_thread = extract_clock_feats(clock)

	# add months since lstg start
	lstg_start = lookup['start_date'] * 24 * 3600
	months = (threads['start_time'] - lstg_start) / (3600 * 24 * MAX_DAYS)
	x_thread.loc[:, 'months_since_lstg'] = months

	# add buyer history deciles
	hist = threads['byr_hist']
	hist.loc[:, hist == 1] -= 1e-16
	hist = np.floor(HIST_QUANTILES * hist) / HIST_QUANTILES
	x_thread.loc[:, 'byr_hist'] = hist

	# save
	dump(x_thread, path('x_thread'))