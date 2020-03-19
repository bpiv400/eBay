import argparse
import numpy as np
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from processing.e_inputs.inputs_utils import get_y_con, get_y_msg
from processing.processing_utils import concat_sim_chunks, load_file
from processing.processing_consts import NUM_OUT
from constants import TEST
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


def main():
	# extract model outcome from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=True, type=str)
	name = parser.parse_args().name

	# lookup
	lookup = load_file(TEST, 'lookup')

	# number of periods
	num_out = NUM_OUT[name]
	if num_out == 1:
		num_out += 1

	# simulated outcomes
	threads, offers = concat_sim_chunks(TEST)
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

	# average model prediction
	p_hat, _ = get_model_predictions(name, data)
	p_hat = p_hat.mean(axis=0)
	assert np.abs(p_hat.sum() - 1) < 1e-8

	# print number of observations
	print('Observations: {} in data/model, {} in sim'.format(
		len(y_obs), len(y_sim)))

	# print comparison
	print('-----------------')
	print('interval: data | model | sim:')
	for i in range(len(p_sim)):
		print('{:3.0f}: {:2.2%} | {:2.2%} | {:2.2%}'.format(
			i, p_obs[i], p_hat[i], p_sim[i]))


if __name__ == '__main__':
	main()
