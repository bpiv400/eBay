import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from compress_pickle import load
from assess.first_arrival.first_arrival_utils import get_batches, PartialDataset
from constants import *
from utils import load_model, load_featnames


def compare_data_model(name):
	# load data
	print('Loading data')
	d = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, name))
	idx = load(INDEX_DIR + '{}/{}.gz'.format(VALIDATION, name))
	featnames = load_featnames(name)

	# reconstruct y
	y = pd.Series(d['y'], index=idx, name='interval')

	# reconstruct x
	x = {}
	for k, v in d['x'].items():
		x[k] = pd.DataFrame(v, index=idx, 
			columns=featnames['offer' if 'offer' in k else k])

	# drop turn indicators from x['lstg']
	cols = [c for c in x['lstg'].columns if c.startswith('t') and len(c) == 2]
	x['lstg'].drop(cols, axis=1, inplace=True)

	# restrict to first thread
	y = y.xs(1, level='thread')
	x = {k: v.xs(1, level='thread').values for k, v in x.items()}

	# create dataset
	data = PartialDataset(x)

	# create model
	net = load_model(name).to('cuda')

	# multinomial
	print('Prediction from model')
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

	# print comparison
	for i in range(len(p)):
		print('\tInterval {}: {:.2%} in data, {:.2%} in model'.format(
			i, (y == i).mean(), p[i]))


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