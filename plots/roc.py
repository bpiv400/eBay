from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import input_fontsize, line_plot
from plots.plots_consts import ROC_PTS
from constants import INPUT_DIR, PLOT_DIR, TEST, DISCRIM_MODELS


def get_auc(s):
	fp = s.index.values
	fp_delta = fp[1:] - fp[:-1]
	tp = s.values
	tp_bar = (tp[1:] + tp[:-1]) / 2
	auc = (fp_delta * tp_bar).sum()
	return auc


def get_roc(p, y):
	s = pd.Series(name='true_positive_rate')
	for i in range(ROC_PTS + 1):
		y_hat = p > float(i / ROC_PTS)
		fp = (~y & y_hat).sum() / (~y).sum()
		tp = (y & y_hat).sum() / y.sum()
		s.loc[fp] = tp
	# check for doubles
	assert len(s.index) == len(s.index.unique())
	return s.sort_index()


def main():
	# extract parameters from command line
	fontsize = input_fontsize()

	for m in DISCRIM_MODELS:
		name = 'p_{}.pkl'.format(m)

		# load data
		p = load(PLOT_DIR + name)
		y = load(INPUT_DIR + '{}/{}.gz'.format(TEST, m))['y']
		assert len(p) == len(y)

		# roc
		s = get_roc(p, y)

		# roc plot
		roc_plot(name, s, fontsize)

		# auc
		print('{}: {}'.format(m, get_auc(s)))
		

if __name__ == '__main__':
    main()