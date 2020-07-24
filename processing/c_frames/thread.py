from compress_pickle import dump
import numpy as np
import pandas as pd
from constants import HIST_QUANTILES, PARTS_DIR
from featnames import BYR_HIST, MONTHS_SINCE_LSTG, START_DATE
from processing.util import get_lstgs, load_feats
from utils import get_months_since_lstg, input_partition


def create_x_thread(lstgs=None):
	# load data
	offers = load_feats('offers', lstgs=lstgs)
	thread_start = offers.clock.xs(1, level='index')
	byr_hist = load_feats('threads', lstgs=lstgs)[BYR_HIST]
	start_date = load_feats('listings', lstgs=lstgs)[START_DATE]
	lstg_start = start_date.astype('int64') * 24 * 3600

	# months since lstg start
	months = get_months_since_lstg(lstg_start, thread_start)
	months = months.rename(MONTHS_SINCE_LSTG)
	assert months.max() < 1

	# buyer history deciles
	hist = np.floor(HIST_QUANTILES * byr_hist).astype('int8')
	assert hist.max() == HIST_QUANTILES - 1

	# create dataframe
	x_thread = pd.concat([months, hist], axis=1)

	return x_thread


def main():
	part = input_partition()
	print('{}/x_thread'.format(part))

	x_thread = create_x_thread(lstgs=get_lstgs(part))
	dump(x_thread, PARTS_DIR + '{}/x_thread.gz'.format(part))


if __name__ == "__main__":
	main()
