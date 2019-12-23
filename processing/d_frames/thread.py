from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import HIST_QUANTILES, PARTS_DIR, CLEAN_DIR, MAX_DAYS
from utils import input_partition
from processing.processing_utils import get_partition


if __name__ == "__main__":
	# data partition
	part = input_partition()
	idx, path = get_partition(part)

	# load data
	lstg_start = load('{}{}/lookup.gz'.format(PARTS_DIR, part)).start_time
	thread_start = load('{}offers.pkl'.format(CLEAN_DIR)).reindex(
		index=idx, level='lstg').clock.xs(1, level='index')
	byr_hist = load('{}threads.pkl'.format(CLEAN_DIR)).reindex(
		index=idx, level='lstg').byr_hist

	# months since lstg start
	months = (thread_start - lstg_start) / (3600 * 24 * MAX_DAYS)
	months = months.rename('months_since_lstg')
	assert months.max() < 1

	# buyer history deciles
	hist = np.floor(HIST_QUANTILES * threads.byr_hist).astype('int8')
	assert hist.max() == 9

	# create dataframe
	x_thread = pd.concat([months, hist], axis=1)

	# save
	dump(x_thread, path('x_thread'))
