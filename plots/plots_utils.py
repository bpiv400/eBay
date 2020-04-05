from constants import FIGURE_DIR

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', **{'family':'serif', 
			      'serif':['Computer Modern Roman'], 
                  'monospace': ['Computer Modern Typewriter']})


def save_fig(name):
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.savefig(FIGURE_DIR + '{}.png'.format(name), 
				format='png', 
				transparent=True,
				bbox_inches='tight')


def draw_lines():
	raise NotImplementedError()


def draw_bars():
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