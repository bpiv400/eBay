import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset, get_sim_times
from processing.processing_utils import load_file, get_arrival_times
from utils import load_model, load_featnames
from constants import VALIDATION, ENV_SIM_DIR, SIM_CHUNKS, INPUT_DIR, INDEX_DIR


def get_sim_hist(thread):
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		s = sim['threads'].byr_hist
		if thread is not None:
			s = s.xs(thread, level='thread')
		y_sim.append(s)
	return pd.concat(y_sim, axis=0).values


def get_sim_con(thread, index):
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		s = sim['offers'].con
		if thread is not None:
			s = s.xs(thread, level='thread')
		if index is not None:
			s = s.xs(index, level='index')
		y_sim.append(s)
	return pd.concat(y_sim, axis=0).values


def get_sim_outcomes(name, thread, index):
	if name == 'arrival':
		lstg_start = load_file(VALIDATION, 'lookup').start_time
		lstg_end, thread_start = get_sim_times(VALIDATION, lstg_start)
		clock = get_arrival_times(lstg_start, lstg_end, thread_start)
		y_sim, _ = get_interarrival_period(clock)
		if thread is not None:
			y_sim = y_sim.xs(thread, level='thread')
		return y_sim

	if name == 'hist':
		return get_sim_hist(thread)

	if name == 'con_byr':
		return get_sim_con(thread, index)


def compare_data_model(name, thread, index):
	# simulations
	print('Loading simulated outcomes')
	y_sim = get_sim_outcomes(name, thread, index)

	# load data
	print('Loading data')
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

	# create dataset
	data = PartialDataset(x)

	# create model
	net = load_model(name).to('cuda')

	# multinomial
	print('Generating predictions from model')
	batches = get_batches(data)
	p = None
	for b in batches:
		x_b = {k: v.to('cuda') for k, v in b.items()}
		theta = net(x_b)
		p_b = torch.exp(F.log_softmax(theta, dim=1)).cpu().numpy()
		if p is None:
			p = p_b
		else:
			p = np.append(p, p_b, axis=0)

	# take average
	p = p.mean(axis=0)
	assert y.max() + 1 == len(p)

	# print number of observations
	print('-----------------')
	print('Observations: {} in data/model, {} in sim'.format(
		len(y), len(y_sim)))
	
	# print comparison
	print('interval: data | model | sim:')
	for i in range(len(p)):
		print('{:3.0f}: {:2.2%} | {:2.2%} | {:2.2%}'.format(
			i, (y == i).mean(), p[i], (y_sim == i).mean()))


def main():
	# extract model name from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=True, type=str)
	parser.add_argument('--thread', required=False, type=int)
	parser.add_argument('--index', required=False, type=int)
	args = parser.parse_args()
	name, thread, index = args.name, args.thread, args.index

	# error checking inputs
	assert name in ['arrival', 'hist', 'con_byr']
	if name in ['arrival', 'hist']:
		assert index is None

	print('Model: {}'.format(name))
	print('Thread: {}'.format(thread))
	print('Index: {}'.format(index))
	compare_data_model(name, thread, index)


if __name__ == '__main__':
	main()