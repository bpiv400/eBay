import argparse
import torch
from torch.nn.functional import log_softmax, nll_loss
import numpy as np
import pandas as pd
from compress_pickle import load
from train.Sample import get_batches
from train.EBayDataset import EBayDataset
from utils import load_model, load_featnames
from constants import VALIDATION, INPUT_DIR, INDEX_DIR, DISCRIM_MODELS


def get_model_predictions(net, data):
	# loop over batches
	print('Predictions from model')
	lnL = []
	batches = get_batches(data)
	for b in batches:
		b['x'] = {k: v.to('cuda') for k, v in b['x'].items()}	
		b['y'] = b['y'].to('cuda')
		theta = net(b['x'])
		if theta.size()[1] == 1:
			theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
		lnq = log_softmax(theta, dim=-1)
		lnL.append(-nll_loss(lnq, b['y'], reduction='none').cpu().numpy())
	return np.exp(np.concatenate(lnL, axis=0))


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='model name')
	name = parser.parse_args().name

	# initialize neural net
	net = load_model(name).to('cuda')

	# create dataset
	print('Loading data')
	data = EBayDataset(VALIDATION, name)

	# model predictions
	p = get_model_predictions(net, data)

	# for discrim models, report accuracy
	if name in DISCRIM_MODELS:
		print('Accuracy: {}'.format((p > 0.5).mean()))

		# correct and certain
		correct = p > 0.99
		idx_obs = np.argwhere(correct & data.d['y'])
		idx_sim = np.argwhere(correct & ~data.d['y'])
		for k in data.d['x'].keys():
			obs = data.d['x'][k][idx_obs,:].mean(axis=0)
			sim = data.d['x'][k][idx_sim,:].mean(axis=0)
			print('{}: {}'.format(k, obs / sim))

	# for other models, report log-likelihood
	else:
		print('Mean log-likelihood: {}'.format(np.mean(lnL)))
		print('Worst log-likelihood: {}'.format(np.min(lnL)))
		for i in [-10, -20, -30]:
			print('Share below {}: {}'.format(i, np.mean(lnL < i)))


if __name__ == '__main__':
	main()
