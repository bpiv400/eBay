from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import roc_plot, input_fontsize
from constants import PLOT_DIR, DISCRIM_MODELS


def get_auc(s):
	fp = s.index.values
	fp_delta = fp[1:] - fp[:-1]
	tp = s.values
	tp_bar = (tp[1:] + tp[:-1]) / 2
	auc = (fp_delta * tp_bar).sum()
	return auc


def get_roc(p):
	s = pd.Series(name='true_positive_rate')
	for i in range(101):
		y_hat = p > float(i / 100)
		fp = (~y & y_hat).sum() / (~y).sum()
		tp = (y & y_hat).sum() / y.sum()
		s.loc[fp] = tp

	# sort
	s = s.sort_index()
	assert len(s.index) == len(s.index.unique())
		
	return s


def main():
	# extract parameters from command line
	fontsize = input_fontsize()

	for m in DISCRIM_MODELS:
		name = 'roc_{}.pkl'.format(m)

		# load data
		p = load(PLOT_DIR + name)[:, 1]

		# roc
		s = get_roc(p)

		# roc plot
		roc_plot(name, s, fontsize)

		# auc
		print('{}: {}'.format(m, get_auc(s)))
		

if __name__ == '__main__':
    main()