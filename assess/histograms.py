import sys, os
import numpy as np, pandas as pd
from compress_pickle import load, dump
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from assess.assess_utils import *
from constants import *


def create_histograms(y, p_hat):
	# array of bins
	bins = np.arange(101)

	# initialize figure
	plt.ion()
	plt.show()

	# one subfigure for each of first 6 turns
	fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)

	for i in range(3):
		for j in range(2):
			# turn number
			turn = i * 2 + j + 1

			# slice by turn number
			y_t = y.xs(turn, level='index')
			p_hat_t = p_hat.xs(turn, level='index')

			# (average) probabilities
			p0 = np.array([np.mean(y_t == b) for b in bins])
			p1 = np.mean(p_hat_t, axis=0)

			# plot subfigure
			axs[i,j].bar(bins, p0, alpha=0.5, label='Observed')
			axs[i,j].bar(bins, p1, alpha=0.5, label='Predicted')

	# legend
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center')


# observed and predicted outcomes
y, p_hat = get_outcomes('con')

# lookup file
L = load(CLEAN_DIR + 'listings.pkl')[['meta', 'start_price_pctile']]
L = L.reindex(index=y.index, level='lstg')

# overall histograms
create_histograms(y, p_hat)

# by meta
s = L.meta
v = [m for m in s.unique() if (s == m).sum() >= 100000]
v = np.sort(np.array(v))
for m in v:
	idx = s[s == m].index
	y_m = y.reindex(index=idx)
	p_hat_m = p_hat.reindex(index=idx)
	create_histograms(y_m, p_hat_m)

# by start price percentile
s = L.start_price_pctile
for i in range(10):
	l = i / 10
	h = (i+1) / 10
	idx = s[(s > l) & (s <= h)].index
	y_i = y.reindex(index=idx)
	p_hat_i = p_hat.reindex(index=idx)
	create_histograms(y_i, p_hat_i)