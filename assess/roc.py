import numpy as np
import pandas as pd
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS, PLOT_DIR


def main():
	# loop over discriminator models
	for m in DISCRIM_MODELS:
		print(m)

		# initialize dataset
		data = EBayDataset(TEST, m)
		y = data.d['y']

		# model predictions
		p, _ = get_model_predictions(m, data)
		p = p[:, 1]

		# ROC curve
		s = pd.Series(name='true_positive_rate')
		for i in range(101):
			y_hat = p > float(i / 100)
			fp = (~y & y_hat).sum() / (~y).sum()
			tp = (y & y_hat).sum() / y.sum()
			s.loc[fp] = tp

		# sort
		s = s.sort_index()
		assert len(s.index) == len(s.index.unique())

		# save dictionary
		dump(s, PLOT_DIR + 'roc_{}.pkl'.format(m))


if __name__ == '__main__':
	main()
