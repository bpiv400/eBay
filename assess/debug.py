import argparse
import numpy as np
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='model name')
	name = parser.parse_args().name

	# initialize dataset
	print('Loading data')
	data = EBayDataset(TEST, name)
	x, y = [data.d[k] for k in ['x', 'y']]

	# model predictions
	p, lnL = get_model_predictions(name, data)

	# for discrim models, report accuracy
	if name in DISCRIM_MODELS:
		print('Accuracy: {}'.format((p > 0.5).mean()))

		# correct and certain
		correct = p > 0.99
		idx_obs = np.argwhere(correct & y)
		idx_sim = np.argwhere(correct & ~y)
		for k in x.keys():
			obs = x[k][idx_obs, :].mean(axis=0)
			sim = x[k][idx_sim, :].mean(axis=0)
			print('{}: {}'.format(k, obs / sim))

	# for other models, report log-likelihood
	else:
		print('Mean log-likelihood: {}'.format(np.mean(lnL)))
		print('Worst log-likelihood: {}'.format(np.min(lnL)))
		for i in [-10, -20, -30]:
			print('Share below {}: {}'.format(i, np.mean(lnL < i)))


if __name__ == '__main__':
	main()
