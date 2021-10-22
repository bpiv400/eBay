import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot.const import FONTSIZE, FIG_DIR

plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = True


def save_fig(path, legend=True, legend_kwargs=None, reverse_legend=False,
             xlabel=None, ylabel=None, square=True, xaxis=True, yaxis=True,
             xticks=None, yticks=None, xticklabels=None, yticklabels=None,
             xlim=None, ylim=None, gridlines=True, integer_xticks=False,
             logx=False, logy=False, legend_outside=False, fontsize=FONTSIZE):

    # axis limits
    if xlim is not None:
        plt.gca().set_xbound(xlim)
    if ylim is not None:
        plt.gca().set_ybound(ylim)

    # legend
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()

        if reverse_legend:
            handles = reversed(handles)
            labels = reversed(labels)

        if legend_kwargs is None:
            legend_kwargs = dict()
        if legend_outside:
            legend_kwargs['bbox_to_anchor'] = (1, 1)
            legend_kwargs['loc'] = 'upper left'
            legend_kwargs['frameon'] = False
        else:
            legend_kwargs['frameon'] = gridlines
        plt.legend(handles=handles,
                   labels=labels,
                   handlelength=1.,
                   fancybox=False,
                   fontsize=fontsize,
                   title_fontsize=fontsize,
                   framealpha=0,
                   **legend_kwargs)

    # axis labels and tick labels
    formatter = FuncFormatter(remove_leading_zero)
    if xlabel is not None:
        plt.xticks(fontsize=fontsize)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xlabel(xlabel, fontsize=fontsize)
    else:
        plt.xticks([])

    if ylabel is not None:
        plt.yticks(fontsize=fontsize)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.ylabel(ylabel, fontsize=fontsize)
    else:
        plt.yticks([])

    if xticks is not None:
        plt.gca().set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    if integer_xticks:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticklabels is not None:
        plt.gca().set_xticklabels(xticklabels)
    if yticklabels is not None:
        plt.gca().set_yticklabels(yticklabels)

    # gridlines
    if gridlines:
        plt.grid(axis='both',
                 which='both',
                 color='gray',
                 linestyle='-',
                 linewidth=0.5)
    else:
        plt.grid(axis='both', visible=False)

    # hide axes
    if not xaxis or not yaxis:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
    if not xaxis:
        plt.gca().spines['bottom'].set_visible(False)
    if not yaxis:
        plt.gca().spines['left'].set_visible(False)

    if square:
        plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(),
                             adjustable='box')

    # save
    plt.savefig(FIG_DIR + '{}.png'.format(path),
                format='png',
                # transparent=True,
                bbox_inches='tight')

    # close
    plt.close()


def remove_leading_zero(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4).
    :param x: value to reformat
    :param pos: unused
    :return: string
    """
    val_str = '{:g}'.format(x)
    if 0 < np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def get_name(path):
    return path.split('/')[-1].split('_')[1]
