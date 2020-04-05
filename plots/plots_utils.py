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

def line_plot(name, x, y, styles, fontsize=16):
	# y and styles must be of same length
	assert len(y) == len(styles)

	# overall plot settings
	plt.clf()

	# loop over lines to draw
	for i in range(len(y)):
		plt.plot(x, y[i], styles[i])

	# axis options
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	# save
	save_fig(name)