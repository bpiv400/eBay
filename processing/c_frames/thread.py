from compress_pickle import load, dump
import numpy as np
import pandas as pd
from constants import HIST_QUANTILES, PARTS_DIR, CLEAN_DIR
from featnames import BYR_HIST, MONTHS_SINCE_LSTG, START_TIME
from util import get_months_since_lstg, input_partition, load_file


def main():
	# data partition
	part = input_partition()
	print('{}/x_thread'.format(part))

	# load data
	lookup = load_file(part, 'lookup')
	thread_start = load(CLEAN_DIR + 'offers.pkl').reindex(
		index=lookup.index, level='lstg').clock.xs(1, level='index')
	byr_hist = load(CLEAN_DIR + 'threads.pkl').reindex(
		index=lookup.index, level='lstg')[BYR_HIST]

	# months since lstg start
	months = get_months_since_lstg(lookup[START_TIME], thread_start).rename(MONTHS_SINCE_LSTG)
	assert months.max() < 1

	# buyer history deciles
	hist = np.floor(HIST_QUANTILES * byr_hist).astype('int8')
	assert hist.max() == 9

	# create dataframe
	x_thread = pd.concat([months, hist], axis=1)

	# save
	dump(x_thread, PARTS_DIR + '{}/x_thread.gz'.format(part))


if __name__ == "__main__":
	main()
