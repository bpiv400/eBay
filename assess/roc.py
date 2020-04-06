import numpy as np
import pandas as pd
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS, PLOT_DIR


def main():
	# initialize output dictionary
	d = dict()

	# loop over discriminator models
	for m in DISCRIM_MODELS:
		# initialize dataset
		data = EBayDataset(TEST, m)
		y = data.d['y']

		# model predictions
		p, _ = get_model_predictions(m, data)
		p = p[:, 1]

		# ROC curve
		d[m] = pd.Series(name='true_positive_rate')
		for i in range(101):
			y_hat = p > float(i / 100)
			fp = (~y & y_hat).sum() / (~y).sum()
			tp = (y & y_hat).sum() / y.sum()
			d[m].loc[fp] = tp

		# sort
		d[m] = d[m].sort_index()
		assert len(d[m].index) == len(d[m].index.unique())

	# save dictionary
	dump(d, PLOT_DIR + 'roc.pkl')


if __name__ == '__main__':
	main()
