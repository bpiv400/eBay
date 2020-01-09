import numpy as np, pandas as pd
from processing.processing_utils import input_partition, save_files, \
	load_file, shave_floats, get_idx_x


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	# thread features
	x_offer = load_file(part, 'x_offer').xs(1, level='index')
	x_offer = x_offer.drop(['days', 'delay', 'con', 'norm', 'split', \
		'msg', 'reject', 'auto', 'exp'], axis=1)
	x_offer = x_offer.rename(lambda x: x + '_1', axis=1)
	x_thread = load_file(part, 'x_thread').join(x_offer)

	# outcome
	y = x_thread['byr_hist']
	idx = y.index
	x_thread.drop('byr_hist', axis=1, inplace=True)

	# shave floats
	x_thread = shave_floats(x_thread)

	# index of listing features
	idx_x = get_idx_x(part, idx)

	return {'y': y, 'x_thread': x_thread, 'idx_x': idx_x}


if __name__ == '__main__':
	# partition name from command line
	part = input_partition()
	print('%s/hist' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, 'hist')
