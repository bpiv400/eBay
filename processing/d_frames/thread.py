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
	events = load(CLEAN_DIR + 'offers.pkl').reindex(
		index=idx, level='lstg')

	# initialize x_thread with calendar features
	thread_start = events.clock.xs(1, level='index')
	clock = pd.to_datetime(thread_start, unit='s', origin=START)
	x_thread = extract_clock_feats(clock)

	# add months since lstg start
	lstg_start = lookup['start_date'].astype('int64') * 24 * 3600
	months = (thread_start - lstg_start) / (3600 * 24 * MAX_DAYS)
	assert months.max() < 1
	x_thread.loc[:, 'months_since_lstg'] = months

	# add buyer history deciles
	hist = threads['byr_hist']
	hist = np.floor(HIST_QUANTILES * hist) / HIST_QUANTILES
	assert hist.max() < 1
	x_thread.loc[:, 'byr_hist'] = hist

	# save
	dump(x_thread, path('x_thread'))
