import os
from constants import FIG_DIR
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FONTSIZE = {'roc': 16,  # fontsize by plot type
            'training': 24,
            'num': 16,
            'sale': 16,
            'response': 16}

plt.style.use('ggplot')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def save_fig(path, legend=True, legend_kwargs=None,
             xlabel=None, ylabel=None, square=False):
    # create directory
    folder, name = path.split('/')
    if not os.path.isdir(FIG_DIR + folder):
        os.mkdir(FIG_DIR + folder)

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
    plt.savefig(FIG_DIR + '{}.png'.format(path),
                format='png',
                # transparent=True,
                bbox_inches='tight')

    # close
    plt.close()


def line_plot(df, style=None, ds='steps-post', xlim=None, ylim=None,
              xticks=None, yticks=None, diagonal=False,
              integer_xticks=False, logx=False):
    plt.clf()

    df.plot.line(style=style,
                 xlim=xlim,
                 ylim=ylim,
                 xticks=xticks,
                 yticks=yticks,
                 ds=ds,
                 legend=False)

    if integer_xticks:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if logx:
        plt.xscale('log')

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


def training_plot(path, df):
    line_plot(df,
              style={'testing': '-k', 'train': '--k', 'baserate': '-k'},
              integer_xticks=True,
              ds='default')
    save_fig(path, legend=False)


def continuous_pdf(path, df, xlabel=None, ylabel=None, **plotargs):
    line_plot(df, **plotargs)
    save_fig(path, xlabel=xlabel, ylabel=ylabel)


def cdf_plot(path, df, xlabel=None, ylabel=None,
             legend_kwargs=None, **plotargs):
    if legend_kwargs is None:
        legend_kwargs = {'loc': 'lower right'}
    line_plot(df, ylim=[0, 1], **plotargs)
    save_fig(path,
             xlabel=xlabel,
             ylabel=ylabel,
             legend_kwargs=legend_kwargs)


def survival_plot(path, df, xlabel=None, ylabel=None, **plotargs):
    line_plot(df, ylim=[0, 1], **plotargs)
    save_fig(path, xlabel=xlabel, ylabel=ylabel,
             legend_kwargs={'loc': 'upper right'})


def roc_plot(path, s):
    line_plot(s, xlim=[0, 1], ylim=[0, 1], diagonal=True, ds='default')
    save_fig(path,
             xlabel='False positive rate',
             ylabel='True positive rate',
             square=True,
             legend=False)


def response_plot(path, df, byr=False):
    line_plot(df, xlim=[.4, .95], ylim=[0, 1], ds='default')
    save_fig(path,
             xlabel='{} offer as fraction of list price'.format(
                 'Seller' if byr else 'Buyer'))


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


def grouped_bar(path, df, xlabel=None, ylabel=None, **plotargs):
    bar_plot(df, **plotargs)
    save_fig(path, xlabel=xlabel, ylabel=ylabel)
