from compress_pickle import load
import numpy as np
from plots.plots_utils import line_plot, save_fig
from constants import PLOT_DIR, MODELS


def main():
	# plot styles
	style = {'test': '-k', 'train': '--k', 'baserate': '-k'}

	# load data
	lnl = load(PLOT_DIR + 'lnL.pkl')

	# loop over models, create plot
	for m in MODELS:
		print(m)

		# line data
		y = dict()
		for k in ['test', 'train', 'baserate']:
			if k in lnl:
				y[k] = np.exp(lnl[m][k])

		num = len(y['test'])
		y[k] = np.repeat(y[k], num)
		
		# make plot
		line_plot(range(num), y, style)

		# save
		name = 'likelihood_{}'.format(m)
		save_fig(name)


if __name__ == '__main__':
	main()
