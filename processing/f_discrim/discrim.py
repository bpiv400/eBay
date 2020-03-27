from compress_pickle import load
import numpy as np
import pandas as pd
from processing.processing_utils import load_file, init_x, get_norm
from processing.f_discrim.discrim_utils import concat_sim_chunks, save_discrim_files
from utils import 
from processing.processing_consts import NUM_OUT
from constants import INPUT_DIR, INDEX_DIR
from featnames import DAYS, DELAY, CON, NORM, SPLIT, REJECT, MSG


def construct_x(x, idx_thread):
	d = x.copy()
	idx = x['lstg'].index
	has_thread = pd.Series(1.0, index=idx_thread, name='has_thread').reindex(
		index=idx, fill_value=0.0)
	d['lstg'] = d['lstg'].join(has_thread)
	return d


def combine_xy(part, name, d):
	x = d['x']
	y = (d['y'] / NUM_OUT[name]).astype('float32')
	if name in ARRIVAL_MODELS:
		x['lstg'] = np.concatenate(x['lstg'], y)
	else:
		featnames = load(INPUT_DIR + 'featnames/{}.pkl'.format(name))['offer']
		key = 'offer' + name[-1]
		if CON in part:
			idx = load(INDEX_DIR + '{}/{}.gz'.format(part, name))
			df = pd.DataFrame(x[key], index=idx, columns=featnames)
			df.loc[:, CON] = y
			df.loc[:, REJECT] = df[CON] == 0
			df.loc[:, NORM] = get_norm(df[CON])
			df.loc[:, SPLIT] = df[CON].apply(is_split)
			x[key] = df.astype('float32').values
		elif MSG in part: 
			x[key][:, featnames.index(MSG)] = y

	return x


def main():
	# extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    part, name = args.part, args.name
	print('discrim/{}/{}}'.format(part, name))

	# observed data
	d_obs = load(INPUT_DIR + '{}/{}.gz'.format(part, name))
	x_obs = combine_xy(part, name, d_obs)

	idx_obs = load_file(part, 'x_thread').xs(1, level='thread').index
	x_obs = construct_x(x, idx_obs)

	# simulated data
	threads_sim, _ = concat_sim_chunks(part, keep_tf=False)
	idx_sim = threads_sim.xs(1, level='thread').index
	x_sim = construct_x(x, idx_sim)

	# save data
	save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
	main()
