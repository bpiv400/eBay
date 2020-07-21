from compress_pickle import dump
import numpy as np
import pandas as pd
from sim.EBayDataset import EBayDataset
from assess.util import get_model_predictions
from assess.const import ROC_DIM
from constants import TEST, DISCRIM_THREADS, PLOT_DIR


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
	# initialize dataset
	data = EBayDataset(TEST, DISCRIM_THREADS)
	y = data.d['y']

	# model predictions
	p, _ = get_model_predictions(DISCRIM_THREADS, data)
	p = p[:, 1]
	p0, p1 = p[y == 0], p[y == 1]

	# calculate roc curve
	roc = get_roc(p0, p1).rename('roc')

	# save predictions
	dump(roc, PLOT_DIR + 'roc.pkl')


if __name__ == '__main__':
	main()
