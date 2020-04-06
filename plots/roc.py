from compress_pickle import load
import numpy as np
from plots.plots_utils import line_plot, input_fontsize
from constants import PLOT_DIR, DISCRIM_MODELS


def get_auc(s):
	fp = s.index.values
	fp_delta = fp[1:] - fp[:-1]
	tp = d[m].values
	tp_bar = (tp[1:] + tp[:-1]) / 2
	auc = (fp_delta * tp_bar).sum()


def main():
	# extract parameters from command line
	fontsize = input_fontsize()

	for m in DISCRIM_MODELS:
		# load dictionary
		d = load(PLOT_DIR + 'roc.pkl')

		# auc
		print('{}: {}'.format(m, get_auc(d[m])))

		


if __name__ == '__main__':
    main()