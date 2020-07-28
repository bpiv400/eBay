from processing.util import load_feats
from inputs.util import save_files, get_ind_x
from utils import load_file, input_partition
from constants import BYR_HIST_MODEL
from featnames import CLOCK_FEATS, THREAD_COUNT, BYR_HIST, LOOKUP


def process_inputs(part):
	lstgs = load_file(part, LOOKUP).index
	threads = load_file(part, 'x_thread')
	offers = load_file(part, 'x_offer')

	# thread features
	clock_feats = offers.xs(1, level='index')[CLOCK_FEATS]
	x_thread = clock_feats.join(threads.drop(BYR_HIST, axis=1))
	x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread') - 1
	x = {'thread': x_thread}

	# outcome
	y = load_feats('threads', lstgs=lstgs)[BYR_HIST]
	assert (y.index == x['thread'].index).all()

	# indices for listing features
	idx_x = get_ind_x(lstgs=lstgs, idx=y.index)

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
