import sys, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


if __name__ == "__main__":
	# parameter(s) from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', action='store', type=str, required=True)
    part = parser.parse_args().part
    idx, path = get_partition(part)

	# load data
	lookup = load(PARTS_DIR + '%s/lookup.gz' % part)
	threads = load(CLEAN_DIR + 'threads.pkl').reindex(
		index=idx, level='lstg')
	events = load(CLEAN_DIR + 'offers.pkl').reindex(
		index=idx, level='lstg')

	# months since lstg start
	thread_start = events.clock.xs(1, level='index')
	months = (thread_start - lookup.start_time) / (3600 * 24 * MAX_DAYS)
	months = months.rename('months_since_lstg')
	assert months.max() < 1

	# buyer history deciles
	hist = np.floor(HIST_QUANTILES * threads.byr_hist).astype('int8')
	assert hist.max() == 9

	# create dataframe
	x_thread = pd.concat([months, hist], axis=1)

	# save
	dump(x_thread, path('x_thread'))
