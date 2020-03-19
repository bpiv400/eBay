import argparse
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS


def get_cdf(a):
	a = np.sort(a)
	out = []
	for q in range(101):
		out.append(np.percentile(a, q, interpolation='lower'))
	return np.array(out)


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
	_, lnL = get_model_predictions(name, data)
	p_hat = np.exp(lnL)

	# for discrim models, report accuracy
	if name in DISCRIM_MODELS:
		print('Accuracy: {}'.format((p_hat > 0.5).mean()))

		# correct and certain
		correct = p_hat > 0.99
		idx_obs = np.argwhere(correct & y)
		idx_sim = np.argwhere(correct & ~y)
		diff = dict()
		for k in x.keys():
			num_cols = np.shape(x[k])[-1]
			diff[k] = pd.Series(index=range(num_cols), name=k)
			for c in range(num_cols):
				# sort observed and simulated vectors
				obs = np.sort(x[k][idx_obs, c])
				sim = np.sort(x[k][idx_sim, c])
				# endpoints
				largest = np.maximum(np.max(obs), np.max(sim))
				smallest = np.minimum(np.min(obs), np.min(sim))
				# rescale
				obs = (obs - smallest) / (largest - smallest)
				sim = (sim - smallest) / (largest - smallest)
				assert obs.min() >= 0 and obs.max() <= 1
				assert sim.min() >= 0 and sim.max() <= 1
				# cdfs
				obs_cdf = get_cdf(obs)
				sim_cdf = get_cdf(sim)
				# average absolute difference
				diff[k][c] = np.abs(obs_cdf - sim_cdf).mean()
			print(diff[k])

	# for other models, report log-likelihood
	else:
		print('Mean log-likelihood: {}'.format(np.mean(lnL)))
		print('Worst log-likelihood: {}'.format(np.min(lnL)))
		for i in [-10, -20, -30]:
			print('Share below {}: {}'.format(i, np.mean(lnL < i)))


if __name__ == '__main__':
	main()
