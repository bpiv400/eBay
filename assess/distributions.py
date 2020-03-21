import numpy as np
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from processing.e_inputs.inputs_utils import get_y_con, get_y_msg
from processing.processing_utils import concat_sim_chunks, load_file
from processing.processing_consts import NUM_OUT
from constants import TEST, MODELS
from featnames import BYR_HIST, DELAY, CON, MSG


def get_sim_outcome(name, threads, offers):
	if name == 'first_arrival':
		y_sim = None
	elif name == 'next_arrival':
		y_sim = None
	elif name == 'hist':
		y_sim = threads[BYR_HIST]
	else:
		turn = int(name[-1])
		if name[:-1] == DELAY:
			y_sim = None
		elif name[:-1] == CON:
			y_sim = get_y_con(offers.xs(turn, level='index'))
		elif name[:-1] == MSG:
			y_sim = get_y_msg(offers.xs(turn, level='index'), turn)
		else:
			raise RuntimeError('Invalid name: {}'.format(name))
	return y_sim


def get_distributions(name, threads, offers):
	# number of periods
	num_out = NUM_OUT[name]
	if num_out == 1:
		num_out += 1

	# simulated outcomes
	y_sim = get_sim_outcome(name, threads, offers)

	# average simulated outcome
	p_sim = np.array([(y_sim == i).mean() for i in range(num_out)])
	assert np.abs(p_sim.sum() - 1) < 1e-8

	# observed outcomes
	data = EBayDataset(TEST, name)
	y_obs = data.d['y']

	# average observed outcome
	p_obs = np.array([(y_obs == i).mean() for i in range(num_out)])
	assert np.abs(p_obs.sum() - 1) < 1e-8

	return p_obs, p_sim


def num_threads(df, lstgs):
    s = df.reset_index('thread')['thread'].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)
    s = s.groupby(s).count() / len(lstgs)
    return s


def num_offers(df):
    s = df.reset_index('index')['index'].groupby(['lstg', 'thread']).count()
    s = s.groupby(s).count() / len(s)
    return s


def main():
	# lookup
	lookup = load_file(TEST, 'lookup')

	# simualated outcomes
	threads_sim, offers_sim = concat_sim_chunks(TEST)

	# observed outcomes
	threads_obs = load_file(TEST, 'x_thread')
	offers_obs = load_file(TEST, 'x_offer')

	# remove censored offers
	offers_obs = offers_obs[(offers_obs.delay == 1) | ~offers_obs.exp]

	# number of threads per listing
	num_threads_obs = num_threads(threads_obs, lookup.index)
	num_threads_sim = num_threads(threads_sim, lookup.index)

	# number of offers per thread
	num_offers_obs = num_offers(offers_obs)


	# loop over models, get observed and simulated distributions
	for m in MODELS:
		p_obs, p_sim = get_distributions(m, threads_sim, offers_sim)

	


if __name__ == '__main__':
	main()
