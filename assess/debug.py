import argparse
import torch
import numpy as np
import pandas as pd
from compress_pickle import load
from utils import load_model, load_featnames
from constants import VALIDATION, INPUT_DIR, INDEX_DIR, DISCRIM_MODELS


def get_model_prediction(x, name):
	net = load_model(name)
	x_t = {k: torch.from_numpy(v).float() for k, v in x.items()}
	print('Predictions from model')
	theta = net(x_t)
	return theta


def get_log_likelihood(y, theta):
	# initialize output
	lnL = np.zeros_like(y, dtype='float64')
	# log-probabilities
	if theta.size()[1] == 1:
		theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
	lnp = torch.nn.functional.log_softmax(theta, dim=1).numpy()
	# arrivals
	arrival = y >= 0
	lnL[arrival] = lnp[arrival, y[arrival]]
	# non-arrivals
	if np.min(y) < 0:
		p = np.exp(lnp)
		for i in range(len(y)):
			if y[i] < 0:
				lnL[i] = np.log(np.sum(p[i, y[i]:]))
	return lnL


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='model name')
	name = parser.parse_args().name

	# import data
	print('Loading data')
	d = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, name))

	# model predictions and log-likelihood
	theta = get_model_prediction(d['x'], name)
	lnL = get_log_likelihood(d['y'].astype('int64'), theta)

	# for discrim models, report accuracy
	if name in DISCRIM_MODELS:
		accuracy = np.mean(d['y'] == (theta.numpy() > 0).squeeze())
		print('Accuracy: {}'.format(accuracy))

		# # correct and certain
		# correct = np.exp(lnL) > 0.99
		# idx_obs = np.argwhere(correct & d['y'])
		# idx_sim = np.argwhere(correct & ~d['y'])

	# for other models, report log-likelihood
	else:
		print('Mean log-likelihood: {}'.format(np.mean(lnL)))
		print('Worst log-likelihood: {}'.format(np.min(lnL)))
		for i in [-10, -20, -30]:
			print('Share below {}: {}'.format(i, np.mean(lnL < i)))


if __name__ == '__main__':
	main()
