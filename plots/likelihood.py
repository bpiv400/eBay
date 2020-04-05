from compress_pickle import load
import numpy as np
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
	# load data
	lnL = load(PLOT_DIR + '{}.pkl'.format('lnL'))
	lnL0 = load(PLOT_DIR + '{}.pkl'.format('lnL0'))
	lnL_bar = load(PLOT_DIR + '{}.pkl'.format('lnL_bar'))

	# loop over models, create plot
	ratio = dict()
	for k, v in lnL0.items():
		print(k)
		test = np.exp([v] + lnL[k]['test'])
		train = np.exp([v] + lnL[k]['train'])
		N = len(test)
		epochs = range(N)
		baserate = np.exp(lnL_bar[k])
		ratio[k] = test[-1] / baserate

		# make plot and save
		plt.clf()
		plt.plot(epochs, train, 'k--', 
				 epochs, test, 'k-', 
				 epochs, np.repeat(baserate, N), 'k-')
		save_fig('likelihood/{}.png'.format(k))

	# plot ratios
	for group in [ARRIVAL_MODELS, DELAY_MODELS, CON_MODELS, MSG_MODELS]:
		y = [ratio[m] - 1 for m in group]
		y_pos = np.arange(len(y))
		name = group[0].split('_')[-1]
		if name[-1].isdigit():
			name = name[:-1]

		plt.clf()
		fig, ax = plt.subplots()
		ax.bar(y_pos, y)
		plt.xticks(y_pos, get_model_names(group))
		save_fig('likelihood/baserate_{}.png'.format(name))


if __name__ == '__main__':
    main()
