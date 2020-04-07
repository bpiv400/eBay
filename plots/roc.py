from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import line_plot, save_fig
from plots.plots_consts import ROC_PTS
from constants import INPUT_DIR, PLOT_DIR, TEST, DISCRIM_MODELS


def get_auc(s):
	fp = s.index.values
	fp_delta = fp[1:] - fp[:-1]
	tp = s.values
	tp_bar = (tp[1:] + tp[:-1]) / 2
	auc = (fp_delta * tp_bar).sum()
	return auc


def get_roc(p0, p1):
	s = pd.Series(name='true_positive_rate')
	for i in range(ROC_PTS + 1):
		tau = float(i / ROC_PTS)
		fp = (p0 > tau).sum() / len(p0)
		tp = (p1 > tau).sum() / len(p1)
		s.loc[fp] = tp
	# check for doubles
	assert len(s.index) == len(s.index.unique())
	return s.sort_index()


def main():
	for m in DISCRIM_MODELS:
		print(m)

		# load data
		p0, p1 = load(PLOT_DIR + 'p_{}.pkl'.format(m))

		# roc
		s = get_roc(p0, p1)

		# roc plot
		x = s.index
		y = s.values
		style = '-k'
		line_plot(s.index, s.values, style, diagonal=True)

		# save
		name = 'roc_{}'.format(m)
		save_fig(name, 
				 xlabel='False positive rate',
				 ylabel='True positive rate')

		# auc
		print('{}: {}'.format(m, get_auc(s)))
		

if __name__ == '__main__':
    main()