from inputs.util import save_files, get_ind_x
from utils import load_file, input_partition
from constants import BYR_HIST_MODEL
from featnames import CLOCK_FEATS, THREAD_COUNT, BYR_HIST, LOOKUP


def process_inputs(part):
	lookup = load_file(part, LOOKUP)
	threads = load_file(part, 'x_thread')
	offers = load_file(part, 'x_offer')

	# thread features
	x_offer = offers.xs(1, level='index')
	x_thread = x_offer[CLOCK_FEATS].join(threads)
	x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread') - 1

	# outcome
	y = x_thread[BYR_HIST]
	x = {'thread': x_thread.drop(BYR_HIST, axis=1)}

	# indices for listing features
	idx_x = get_ind_x(lstgs=lookup.index, idx=y.index)

	return {'y': y, 'x': x, 'idx_x': idx_x}


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
