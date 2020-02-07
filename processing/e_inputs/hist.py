import numpy as np
import pandas as pd
from processing.processing_utils import input_partition, load_file, save_files, init_x
from featnames import CLOCK_FEATS, THREAD_COUNT


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	# thread features
	x_offer = load_file(part, 'x_offer').xs(1, level='index')
	x_thread = x_offer[CLOCK_FEATS].join(load_file(part, 'x_thread'))
	x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread')

	# outcome
	y = x_thread['byr_hist']
	idx = y.index
	x_thread = x_thread.drop('byr_hist', axis=1)

	# listing features
	x = init_x(part, idx, drop_slr=True)

	# add thread features to x['lstg']
	x['lstg'] = pd.concat([x['lstg'], x_thread.astype('float32')], axis=1) 

	return {'y': y, 'x': x}


def main():
	# partition name from command line
	part = input_partition()
	print('%s/hist' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, 'hist')


if __name__ == '__main__':
	main()