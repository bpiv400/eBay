from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import roc_plot, input_fontsize
from constants import PLOT_DIR, DISCRIM_MODELS


def get_auc(s):
	fp = s.index.values
	fp_delta = fp[1:] - fp[:-1]
	tp = d[m].values
	tp_bar = (tp[1:] + tp[:-1]) / 2
	auc = (fp_delta * tp_bar).sum()
	return auc


def main():
	# extract parameters from command line
	fontsize = input_fontsize()

	for m in DISCRIM_MODELS:
		name = 'roc_{}.pkl'.format(m)

		# load data
		s = load(PLOT_DIR + name)

		# auc
		print('{}: {}'.format(m, get_auc(s)))

		# roc plot
		roc_plot(name, s, fontsize)


if __name__ == '__main__':
    main()