import pandas as pd
from processing.processing_utils import input_partition, load_file, init_x
from processing.f_discrim.discrim_utils import concat_sim_chunks, \
	get_obs_outcomes, save_discrim_files


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
	print('discrim/{}/listings'.format(part))

	# initialize listing features
	lookup, obs = get_obs_outcomes(part)
	x = init_x(part, lookup.index)

	# observed data
	idx_obs = obs['thread'].xs(1, level='thread').index
	x_obs = construct_x(x, idx_obs)

	# simulated data
	sim = concat_sim_chunks(part)
	idx_sim = sim['threads'].xs(1, level='thread').index
	x_sim = construct_x(x, idx_sim)

	# save data
	save_discrim_files(part, 'listings', x_obs, x_sim)


if __name__ == '__main__':
	main()
