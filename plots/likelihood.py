import argparse
from compress_pickle import load
import numpy as np
from plots.plots_utils import line_plot
from constants import PLOT_DIR, ARRIVAL_MODELS, DELAY_MODELS, CON_MODELS, MSG_MODELS

def get_model_names(models):
	# replace underscores
	models = [m.replace('_', '\_') for m in models]
	if models[0][-1].isdigit():
		names = [r'$\texttt{%s}_%s$' % (m[:-1], m[-1]) for m in models]
	else:
		names = [r'$\texttt{%s}$' % m for m in models]
	return names


def main():
	# extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--fontsize', type=int, default=30)
    args = parser.parse_args()

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
		x = range(N)

		# make plot and save
		name = 'likelihood/{}'.format(k)
		y = [test, train, np.repeat(np.exp(lnL_bar[k]), N)]
		styles = ['-k', '--k', '-k']
		line_plot(name, x, y, styles, fontsize=args.fontsize)


if __name__ == '__main__':
    main()
