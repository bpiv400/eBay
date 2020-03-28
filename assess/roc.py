import numpy as np
import pandas as pd
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS, PLOT_DATA_DIR


def main():
	# initialize output dictionary
	d = dict()

	# loop over discriminator models
	# for m in DISCRIM_MODELS:
	for m in ['threads_no_tf']:
		# initialize dataset
		print('Loading data')
		data = EBayDataset(TEST, m)
		y = data.d['y']

		# model predictions
		p, lnL = get_model_predictions(m, data)
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

		# auc
		fp = d[m].index.values
		fp_delta = fp[1:] - fp[:-1]
		tp = d[m].values
		tp_bar = (tp[1:] + tp[:-1]) / 2
		auc = (fp_delta * tp_bar).sum()
		print('{}: {}'.format(m, auc))

	# save dictionary
	# dump(d, PLOT_DATA_DIR + '{}.pkl'.format('roc_discrim'))


if __name__ == '__main__':
	main()
