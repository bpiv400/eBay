import argparse
import torch
import numpy as np
import pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset
from utils import load_model, load_featnames
from constants import VALIDATION, INPUT_DIR, INDEX_DIR


def import_data(name):
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
	# x to numpy
	x = {k: v.values for k, v in x.items()}
	# create dataset
	data = PartialDataset(x)
	return y.astype('int64'), data


def get_model_prediction(data, net):
	print('predictions from model')
	batches = get_batches(data)
	a = []
	for b in batches:
		x_b = {k: v.to('cuda') for k, v in b.items()}
		theta = net(x_b)
		if theta.size()[1] == 1:
			theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
		lnp = torch.nn.functional.log_softmax(theta, dim=1)
		a.append(lnp.cpu().numpy())
	return np.concatenate(a)


def get_log_likelihood(y, lnp):
	# initialize output
	lnL = pd.Series(0.0, index=y.index)
	# arrivals
	arrival = y >= 0
	lnL[arrival] = lnp[arrival, y[arrival]]
	# non-arrivals
	cens = y < 0
	if np.sum(cens) > 0:
		y_cens = y[cens]
		p_cens = np.exp(lnp[cens, :])
		for i in range(len(y_cens)):
			lnL.loc[y_cens.index[i]] = np.log(np.sum(p_cens[i, y_cens.iloc[i]:]))
	return lnL


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='model name')
	name = parser.parse_args().name

	# load model
	net = load_model(name).to('cuda')

	# import data
	y, data = import_data(name)

	# model predictions and log-likelihood
	lnp = get_model_prediction(data, net)
	lnL = get_log_likelihood(y, lnp)

	# summary statistics
	print('Mean log-likelihood: {}'.format(np.mean(lnL)))
	print('Worst log-likelihood: {}'.format(np.min(lnL)))
	for i in [-10, -20, -30]:
		print('Share below {}: {}'.format(i, np.mean(lnL < i)))


if __name__ == '__main__':
	main()
