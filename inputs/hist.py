import pandas as pd
from inputs.utils import save_files
from util import load_file, input_partition, init_x
from constants import BYR_HIST_MODEL
from featnames import CLOCK_FEATS, THREAD_COUNT, BYR_HIST


# loads data and calls helper functions to construct train inputs
def process_inputs(part):
	threads = load_file(part, 'x_thread')
	offers = load_file(part, 'x_offer')

	# thread features
	x_offer = offers.xs(1, level='index')
	x_thread = x_offer[CLOCK_FEATS].join(threads)
	x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread') - 1

	# outcome
	y = x_thread[BYR_HIST]
	idx = y.index
	x_thread = x_thread.drop(BYR_HIST, axis=1)

	# listing features
	x = init_x(part, idx)

	# add thread features to x['lstg']
	x['lstg'] = pd.concat([x['lstg'], x_thread.astype('float32')], axis=1) 

	return {'y': y, 'x': x}


def main():
	# partition name from command line
	part = input_partition()
	print('{}/{}'.format(part, BYR_HIST_MODEL))

	# input dataframes, output processed dataframes
	d = process_inputs(part)

	# save various output files
	save_files(d, part, BYR_HIST_MODEL)


if __name__ == '__main__':
	main()
