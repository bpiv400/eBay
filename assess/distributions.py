import argparse
import torch
import numpy as np
import pandas as pd
from compress_pickle import load
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from processing.processing_utils import load_file, get_arrival_times, get_interarrival_period
from utils import load_featnames, concat_sim_chunks
from constants import TEST, ENV_SIM_DIR, SIM_CHUNKS, INPUT_DIR, INDEX_DIR


def get_sim_hist(thread):
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		s = sim['threads'].byr_hist
		if thread is not None:
			s = s.xs(thread, level='thread')
		y_sim.append(s)
	return pd.concat(y_sim, axis=0).values


def get_sim_offer(thread, index):
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		df = sim['offers']
		y_sim.append(df)
	return pd.concat(y_sim, axis=0)


def get_sim_outcome(name):
	# collect simulated threads and offers
	threads, offers = concat_sim_chunks(TEST)

	if outcome == 'arrival':
		lstg_start = load_file(VALIDATION, 'lookup').start_time
		lstg_end, thread_start = get_sim_times(VALIDATION, lstg_start)
		clock = get_arrival_times(lstg_start, lstg_end, thread_start)
		y_sim, _ = get_interarrival_period(clock)
		return y_sim

	if name == 'hist':
		return threads.values

	return offers.values


def import_data(name, thread, index):
	d = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, name))
	idx = load(INDEX_DIR + '{}/{}.gz'.format(VALIDATION, name))
	featnames = load_featnames(name)

	# reconstruct y
	y = pd.Series(d['y'], index=idx)

	# reconstruct x
	x = {}
	for k, v in d['x'].items():
		cols = featnames['offer' if 'offer' in k else k]
		x[k] = pd.DataFrame(v, index=idx, columns=cols)

	# restrict to offer index
	if index is not None:
		y = y.xs(index, level='index')
		x = {k: v.xs(index, level='index') for k, v in x.items()}

	# restrict to thread
	if thread is not None:
		y = y.xs(thread, level='thread')
		x = {k: v.xs(thread, level='thread') for k, v in x.items()}

	# x to numpy
	x = {k: v.values for k, v in x.items()}

	return y, x


def get_obs_outcome(name):
	# initialize dataset
	print('Loading data')
	data = EBayDataset(VALIDATION, name)
	x, y = [data.d[k] for k in ['x', 'y']]

	# model predictions
	p, _ = get_model_predictions(name, data)

	# take average
	p = p.mean(axis=0)
	assert y.max() + 1 == len(p)

	return y, p


def compare_data_model(name):
	# simulations
	print('Loading simulated outcomes')
	y_sim = get_sim_outcome(name)

	# load data and get model prediction
	print('Loading data')
	y_obs, p = get_obs_outcome(name)

	# print number of observations
	print('-----------------')
	print('Observations: {} in data/model, {} in sim'.format(
		len(y_obs), len(y_sim)))
	
	# print comparison
	print('interval: data | model | sim:')
	for i in range(len(p)):
		print('{:3.0f}: {:2.2%} | {:2.2%} | {:2.2%}'.format(
			i, (y_obs == i).mean(), p[i], (y_sim == i).mean()))


def main():
	# extract model outcome from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=True, type=str)
	name = parser.parse_args().args

	# error checking inputs
	assert outcome in ['arrival', 'hist', 'delay', 'con', 'msg']
	if outcome in ['arrival', 'hist']:
		assert index is None
	if outcome == 'delay':
		raise NotImplementedError()

	print('Outcome: {}'.format(outcome))
	print('Thread: {}'.format(thread))
	print('Index: {}'.format(index))
	compare_data_model(outcome, thread, index)


if __name__ == '__main__':
	main()