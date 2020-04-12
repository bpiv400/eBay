import numpy as np
from plots.plots_consts import FONTSIZE
from constants import FIGURE_DIR

import matplotlib.pyplot as plt
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.rc('text', usetex=True)
# plt.rc('font', **{'serif': ['Computer Modern Roman'], 
#                   'monospace': ['Computer Modern Typewriter']})


def save_fig(name, legend=None, xlabel=None, ylabel=None, gridlines=True, square=False):
    # font size
    fontsize = FONTSIZE[name.split('_')[0]]

    # aspect ratio
    if square:
        plt.gca().set_aspect(1)

    # tick labels
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # legend
    if legend is not None:
        plt.legend(loc=legend, fontsize=fontsize, fancybox=False)

    # axis labels
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)

    # grid lines
    if gridlines:
        plt.grid(axis='both',
                 which='both',
                 color='gray',
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


def grouped_bar(labels, y):
    plt.clf()

    # styles
    colors = ['k', 'gray']

    # error checking
    assert type(y) is dict and len(y.keys()) == 2

    # parameters
    width = 0.35
    x = np.arange(len(labels))
    num = len(y.keys())

    # create bars
    keys = list(y.keys())
    plt.bar(x - width/num, y[keys[0]], width,
            label=keys[0], color=colors[0])
    plt.bar(x + width/num, y[keys[1]], width,
            label=keys[1], color=colors[1])

    # x labels
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(labels, usetex='$' in labels[0])


def overlapping_bar(y, width=1, alpha=.5, ticks=None, labels=None):
    plt.clf()

    keys = list(y.keys())
    x = np.arange(len(y[keys[0]]))

    # plot bars
    plt.bar(x, y[keys[0]],
            width=width, alpha=alpha, color='white', edgecolor='k')
    plt.bar(x, y[keys[1]],
            width=width, alpha=alpha, color='gray', edgecolor='k')

    # x labels
    if ticks is not None:
        plt.gca().set_xticks(ticks)
        if labels is not None:
            plt.gca().set_xticklabels(labels)
