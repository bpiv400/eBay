import numpy as np
from plots.plots_consts import FIG_DIR, FONTSIZE

import matplotlib.pyplot as plt
plt.style.use('grayscale')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def save_fig(name, legend=True, legend_kwargs=None,
             xlabel=None, ylabel=None, square=False):
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
        if legend_kwargs is None:
            legend_kwargs = dict()
        plt.legend(**legend_kwargs,
                   fancybox=False,
                   frameon=False,
                   fontsize=fontsize)

    # axis labels
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # save
    plt.savefig(FIG_DIR + '{}.png'.format(name),
                format='png',
                transparent=True,
                bbox_inches='tight')

    # close
    plt.close()


def line_plot(df, style=None, ds='steps-post', xlim=None, ylim=None,
              xticks=None, yticks=None, diagonal=False):
    plt.clf()

    df.plot.line(style=style,
                 xlim=xlim,
                 ylim=ylim,
                 xticks=xticks,
                 yticks=yticks,
                 ds=ds,
                 legend=False)

    # gridlines
    plt.grid(axis='both',
             which='both',
             color='gray',
             linestyle='-',
             linewidth=0.5)

    # add 45 degree line
    if diagonal:
        low, high = df.index.min(), df.index.max()
        plt.gca().plot([low, high], [low, high], '--k', linewidth=0.5)


def training_plot(name, df):
    line_plot(df,
              style={'testing': '-k', 'train': '--k', 'baserate': '-k'},
              ds='default')
    save_fig('training_{}'.format(name), legend=False)


def continuous_pdf(name, df, xticks=None, xlabel=None, ylabel=None):
    line_plot(df, xticks=xticks)
    save_fig('p_{}'.format(name), xlabel=xlabel, ylabel=ylabel)


def cdf_plot(name, df, xlim=None, xticks=None, xlabel=None, ylabel=None):
    line_plot(df, xlim=xlim, ylim=[0, 1], xticks=xticks)
    save_fig('p_{}'.format(name),
             xlabel=xlabel,
             ylabel=ylabel,
             legend_kwargs={'loc': 'lower right'})


def survival_plot(name, df, xlim=None, xticks=None, xlabel=None, ylabel=None):
    line_plot(df, xlim=xlim, ylim=[0, 1], xticks=xticks)
    save_fig('p_{}'.format(name), xlabel=xlabel, ylabel=ylabel,
             legend_kwargs={'loc': 'upper right'})


def roc_plot(name, s):
    line_plot(s, xlim=[0, 1], ylim=[0, 1], diagonal=True, ds='default')
    save_fig('roc_{}'.format(name),
             xlabel='False positive rate',
             ylabel='True positive rate',
             square=True,
             legend=False)


def bar_plot(df, horizontal=False, stacked=False, rot=0,
             xlim=None, ylim=None, xticks=None, yticks=None):
    plt.clf()

    # draw bars
    f = df.plot.barh if horizontal else df.plot.bar
    ax = f(stacked=stacked, xlim=xlim, ylim=ylim, legend=False, rot=rot)
    if horizontal:
        ax.invert_yaxis()

    # ticks
    if xticks is not None:
        plt.gca().set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)


def grouped_bar(name, df, horizontal=False, xlabel=None, ylabel=None,
                xlim=None, ylim=None):
    bar_plot(df, horizontal=horizontal, xlim=xlim, ylim=ylim)
    save_fig('p_{}'.format(name), xlabel=xlabel, ylabel=ylabel)
