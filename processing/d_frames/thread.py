from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import HIST_QUANTILES, PARTS_DIR, CLEAN_DIR
from processing.processing_utils import input_partition
from processing.d_frames.frames_utils import get_partition
from processing.processing_consts import CLEAN_DIR, PARTS_DIR
from featnames import BYR_HIST
from utils import get_months_since_lstg


def main():
	# data partition
	part = input_partition()
	idx, path = get_partition(part)
	print('{}/x_thread.gz'.format(part))

	# load data
	lstg_start = load('{}{}/lookup.gz'.format(PARTS_DIR, part)).start_time
	thread_start = load('{}offers.pkl'.format(CLEAN_DIR)).reindex(
		index=idx, level='lstg').clock.xs(1, level='index')
	byr_hist = load('{}threads.pkl'.format(CLEAN_DIR)).reindex(
		index=idx, level='lstg')[BYR_HIST]

	# months since lstg start
	months = get_months_since_lstg(lstg_start, thread_start).rename(MONTHS_SINCE_LSTG)
    assert months.max() < 1

	# buyer history deciles
	hist = np.floor(HIST_QUANTILES * byr_hist).astype('int8')
	assert hist.max() == 9

	# create dataframe
	x_thread = pd.concat([months, hist], axis=1)

	# save
	dump(x_thread, path('x_thread'))


if __name__ == "__main__":
	main()