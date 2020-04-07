import argparse
from plots.plots_consts import GRAY, FONTSIZE
from constants import FIGURE_DIR

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', **{'family':'serif', 
			      'serif':['Computer Modern Roman'], 
                  'monospace': ['Computer Modern Typewriter']})


def save_fig(name, xlabel=None, ylabel=None, gridlines=True, square=True):
	# fontsize
	fontsize = FONTSIZE[name.split('_')[0]]

	# get axes
	fig, ax = plt.subplots()

	# square aspect ratio
	if square:
		ax.set_aspect('equal')

	# tick size
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	# axis labels
	if xlabel is not None:
		plt.xlabel(xlabel, fontsize=fontsize)
	if ylabel is not None:
		plt.ylabel(ylabel, fontsize=fontsize)
	
	# grid lines
	if gridlines:
		plt.grid(axis='both', 
				 which='both', 
				 color=GRAY, 
				 linestyle='-', 
				 linewidth=0.5)

	# save
	plt.savefig(FIGURE_DIR + '{}.png'.format(name), 
				format='png', 
				transparent=True,
				bbox_inches='tight')


def line_plot(x, y, style, diagonal=False):
	# initialize plot
	plt.clf()

	# loop over lines to draw
	if type(y) is list:
		assert len(y) == len(style)
		for i in range(len(y)):
			plt.plot(x, y[i], style[i])
	else:
		plt.plot(x, y, style)

	# add 45 degree line
	if diagonal:
		endpoints = [x.min(), x.max()]
		plt.plot(endpoints, endpoints, '--k', linewidth=0.5)


def input_fontsize():
	parser = argparse.ArgumentParser()
	parser.add_argument('--fontsize', type=int, default=20)
	args = parser.parse_args()
	return args.fontsize