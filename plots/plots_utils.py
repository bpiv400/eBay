import numpy as np
from plots.plots_consts import GRAY, FONTSIZE
from constants import FIGURE_DIR

import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.rc('font', **{'serif': ['Computer Modern Roman'], 
#                   'monospace': ['Computer Modern Typewriter']})


def save_fig(name, legend=False, xlabel=None, ylabel=None, gridlines=True, square=False):
    # font size
    fontsize = FONTSIZE[name.split('_')[0]]

    # aspect ratio
    if square:
        plt.gca().set_aspect(1)

    # tick labels
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # legend
    if legend:
        plt.legend(loc='lower right', fontsize=fontsize, fancybox=False)

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

    # close
    plt.close()


def line_plot(x, y, style, diagonal=False, square=True):
    # initialize plot
    plt.clf()

    # loop over lines to draw
    if type(y) is dict:
        if type(x) is dict:
            assert x.keys() == y.keys()
            for k in y.keys():
                plt.plot(x[k], y[k], style[k], label=k)
        else:
            for k in y.keys():
                plt.plot(x, y[k], style[k], label=k)
    else:
        plt.plot(x, y, style)

    # add 45 degree line
    if diagonal:
        if type(x) is dict:
            low = np.min([a.min() for a in x.values()])
            high = np.max([a.max() for a in x.values()])
        else:
            low, high = x.min(), x.max()
        plt.plot([low, high], [low, high], '--k', linewidth=0.5)


def overlapped_bar(x, y0, y1, width=1, alpha=.5):
    assert len(x) == len(y0) == len(y1)
    num = len(x)
    indices = np.arange(num)

    # plot bars
    plt.bar(x, y0, width=width, alpha=alpha, color='white', edgecolor='k')
    plt.bar(x, y1, width=width, alpha=alpha, color='gray', edgecolor='k')
