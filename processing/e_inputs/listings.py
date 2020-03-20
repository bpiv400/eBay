import pandas as pd
from processing.processing_utils import input_partition, load_file
from processing.e_inputs.inputs_utils import init_x, save_discrim_files
from utils import concat_sim_chunks


def construct_x(x, idx_thread):
	d = x.copy()
	idx = x['lstg'].index
	has_thread = pd.Series(1.0, index=idx_thread, name='has_thread').reindex(
		index=idx, fill_value=0.0)
	d['lstg'] = d['lstg'].join(has_thread)
	return d


def main():
	# extract partition from command line
	part = input_partition()
	print('%s/listings' % part)

	# initialize listing features
	idx = load_file(part, 'lookup').index
	x = init_x(part, idx)

	# observed data
	idx_obs = load_file(part, 'x_thread').xs(1, level='thread').index
	x_obs = construct_x(x, idx_obs)

	# simulated data
	threads_sim, _ = concat_sim_chunks(part, keep_tf=False)
	idx_sim = threads_sim.xs(1, level='thread').index
	x_sim = construct_x(x, idx_sim)

	# save data
	save_discrim_files(part, 'listings', x_obs, x_sim)


if __name__ == '__main__':
	main()
