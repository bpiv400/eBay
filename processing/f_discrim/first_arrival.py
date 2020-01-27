import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset, get_sim_times
from processing.processing_utils import load_file, get_arrival_times, get_interarrival_period
from utils import load_model, load_featnames
from constants import VALIDATION, ENV_SIM_DIR, SIM_CHUNKS, INPUT_DIR, INDEX_DIR


def get_sim_hist():
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		s = sim['threads'].byr_hist.xs(1, level='thread')
		y_sim.append(s)

	return pd.concat(y_sim, axis=0).values


def get_sim_con():
	y_sim = []
	for i in range(1, SIM_CHUNKS+1):
		sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
		s = sim['offers'].con.xs(1, level='thread').xs(1, level='index')
		y_sim.append(s)

	return pd.concat(y_sim, axis=0).values


def get_sim_outcomes(name):
	if name == 'arrival':
		lstg_start = load_file(VALIDATION, 'lookup').start_time
		lstg_end, thread_start = get_sim_times(VALIDATION, lstg_start)
		clock = get_arrival_times(lstg_start, lstg_end, thread_start)
		y_sim, _ = get_interarrival_period(clock)
		y_sim = y_sim.xs(1, level='thread')
		return y_sim

	if name == 'hist':
		return get_sim_hist()

	if name == 'con_byr':
		return get_sim_con()


def compare_data_model(name):
	# simulations
	print('Loading simulated outcomes')
	y_sim = get_sim_outcomes(name)

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
		x[k] = pd.DataFrame(v, index=idx, 
			columns=featnames['offer' if 'offer' in k else k])

	# drop turn indicators from x['lstg']
	cols = [c for c in x['lstg'].columns if c.startswith('t') and len(c) == 2]
	x['lstg'].drop(cols, axis=1, inplace=True)

	# restrict to first offer
	if 'index' in y.index.names:
		y = y.xs(1, level='index')
		x = {k: v.xs(1, level='index') for k, v in x.items()}

	# restrict to first thread
	y = y.xs(1, level='thread')
	x = {k: v.xs(1, level='thread') for k, v in x.items()}

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
	name = parser.parse_args().name
	assert name in ['arrival', 'hist', 'con_byr']

	print('Model: {}'.format(name))
	compare_data_model(name)


if __name__ == '__main__':
	main()