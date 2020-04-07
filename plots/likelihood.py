from compress_pickle import load
import numpy as np
from plots.plots_utils import line_plot, save_fig
from constants import PLOT_DIR


def main():
	# load data
	lnL = load(PLOT_DIR + '{}.pkl'.format('lnL'))
	lnL0 = load(PLOT_DIR + '{}.pkl'.format('lnL0'))
	lnL_bar = load(PLOT_DIR + '{}.pkl'.format('lnL_bar'))

	# loop over models, create plot
	for k, v in lnL0.items():
		print(k)
		test = np.exp([v] + lnL[k]['test'])
		train = np.exp([v] + lnL[k]['train'])
		N = len(test)
		
		# make plot
		x = range(N)
		y = [test, train, np.repeat(np.exp(lnL_bar[k]), N)]
		style = ['-k', '--k', '-k']
		line_plot(x, y, style)

		# save
		name = 'likelihood_{}'.format(k)
		save_fig(name)


if __name__ == '__main__':
    main()
