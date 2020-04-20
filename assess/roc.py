import os
from compress_pickle import dump
import numpy as np
import pandas as pd
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from assess.assess_consts import ROC_DIM
from constants import TEST, DISCRIM_MODELS, PLOT_DIR, MODEL_DIR


def get_roc(p0, p1):
	s = pd.Series()
	for tau in ROC_DIM:
		fp = np.sum(p0 > tau) / len(p0)
		tp = np.sum(p1 > tau) / len(p1)
		s.loc[fp] = tp
	# check for doubles
	assert len(s.index) == len(s.index.unique())
	return s.sort_index()


def main():
	roc = dict()

	# loop over discriminator models
	for m in DISCRIM_MODELS:
		if not os.path.isfile(MODEL_DIR + '{}.net'.format(m)):
			continue

		m = m.replace('_discrim', '')
		print(m)

		# initialize dataset
		data = EBayDataset(TEST, m)
		y = data.d['y']

		# model predictions
		p, _ = get_model_predictions(m, data)
		p = p[:, 1]
		p0, p1 = p[y == 0], p[y == 1]

		# calculate roc curve
		roc[m] = get_roc(p0, p1).rename(m)

	# save predictions
	dump(roc, PLOT_DIR + 'roc.pkl')


if __name__ == '__main__':
	main()
