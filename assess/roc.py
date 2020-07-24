from compress_pickle import dump
import numpy as np
import pandas as pd
from sim.EBayDataset import EBayDataset
from utils import get_model_predictions
from assess.const import ROC_DIM
from constants import TEST, DISCRIM_MODEL, PLOT_DIR


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
	data = EBayDataset(part=TEST, name=DISCRIM_MODEL)
	y = data.d['y']

	# model predictions
	p = get_model_predictions(data)
	p = p[:, 1]
	p0, p1 = p[y == 0], p[y == 1]

	# calculate roc curve
	roc = get_roc(p0, p1).rename('roc')

	# save predictions
	dump(roc, PLOT_DIR + 'roc.pkl')


if __name__ == '__main__':
	main()
