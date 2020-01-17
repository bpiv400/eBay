import numpy as np, pandas as pd
from processing.processing_utils import input_partition, save_files, \
	load_file, init_x
from featnames import CLOCK_FEATS, TIME_FEATS


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	# thread features
	x_offer = load_file(part, 'x_offer').xs(1, level='index')
	x_offer = x_offer[CLOCK_FEATS + TIME_FEATS]
	x_offer = x_offer.rename(lambda x: x + '_1', axis=1)
	x_thread = load_file(part, 'x_thread').join(x_offer)

	# outcome
	y = x_thread['byr_hist']
	idx = y.index

	# listing features
	x = init_x(part, idx)

	# add thread features to x['lstg']
	x_thread = x_thread.drop('byr_hist', axis=1).astype('float32')
	x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1) 

	return {'y': y, 'x': x}


if __name__ == '__main__':
	# partition name from command line
	part = input_partition()
	print('%s/hist' % part)

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, 'hist')
