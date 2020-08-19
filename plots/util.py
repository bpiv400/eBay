import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from constants import FIG_DIR, IDX, BYR, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, \
    DAY, HOUR
from featnames import CON, NORM, DELAY, ARRIVAL, BYR_HIST, MSG

FONTSIZE = {'training': 24}  # fontsize by plot type

plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = True


def save_fig(path, legend=True, legend_kwargs=None, reverse_legend=False,
             xlabel=None, ylabel=None, square=False):
    name = path.split('/')[-1]
    cat = name.split('_')[0]

    # font size
    fontsize = 16 if cat not in FONTSIZE else FONTSIZE[cat]

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
        if reverse_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.gca().legend(reversed(handles), reversed(labels))

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


def dist_plot(path, df):
    names = path.split('/')[-1].split('_')
    dist, name = names[0], '_'.join(names[1:])

    # for y label
    if dist == 'cdf':
        prefix = 'Cumulative share'
    elif dist == 'pdf':
        prefix = 'Share'
    else:
        raise NotImplementedError('Invalid distribution type: {}'.format(dist))

    # labels and plot arguments
    den, ylim = 'listings', [0, 1]
    if name.startswith('price'):
        df = df.loc[df.index > 0, :]
        args = dict(xlim=[1, 1000], logx=True)
        xlabel = 'Sale price'
    elif name.startswith(NORM):
        args = dict(xlim=[0, 1])
        xlabel = 'Sale price / list price'
    elif name.startswith('months'):
        args = dict(xlim=[0, 1])
        xlabel = 'Fraction of month'
    elif name.startswith(ARRIVAL):
        upper = MAX_DELAY_ARRIVAL / DAY
        args = dict(xticks=np.arange(0, upper, 3),
                    xlim=[0, upper], ylim=[0, 0.02])
        xlabel = 'Days since listing start'
        den = 'buyers, by hour'
    elif name.startswith(BYR_HIST):
        args = dict(xlim=[0, 250])
        xlabel = 'Prior best-offer threads for buyer'
        den = 'threads'
    elif name.startswith(DELAY):
        upper = int(MAX_DELAY_TURN / HOUR)
        args = dict(xlim=[0, upper], xticks=np.arange(0, upper + 1e-8, 6))
        xlabel = 'Response time in hours'
        den = 'offers'
    elif name.startswith(CON):
        args = dict(xlim=[0, 100], ylim=[0, 1],
                    xticks=np.arange(0, 100 + 1e-8, 10))
        xlabel = 'Concession (%)'
        den = 'offers'
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # create plot and save
    line_plot(df, **args)
    save_fig(path,
             xlabel=xlabel,
             ylabel='{} of {}'.format(prefix, den),
             legend=isinstance(df, pd.DataFrame))


def training_plot(path, df):
    line_plot(df,
              style={'testing': '-k', 'train': '--k', 'baserate': '-k'},
              integer_xticks=True,
              ds='default')
    save_fig(path, legend=False)


def diag_plot(path, df):
    name = path.split('/')[-1]
    if name.startswith('roc'):
        limits = dict(xlim=[0, 1], ylim=[0, 1])
        labels = dict(xlabel='False positive rate',
                      ylabel='True positive rate')
    elif name.startswith(NORM):
        limits = dict(xlim=[.5, 1.], ylim=[.5, 1.])
        turn = int(name.split('_')[-1])
        last = 'buyer' if turn in [2, 4, 6] else 'seller'
        role = 'buyer' if turn in [3, 5, 7] else 'seller'
        labels = dict(xlabel='Previous {} offer / list price'.format(last),
                      ylabel='Average {} offer / list price'.format(role))
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    line_plot(df, diagonal=True, ds='default', **limits)
    save_fig(path, square=True, legend=isinstance(df, pd.DataFrame), **labels)


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


def count_plot(path, df):
    name = path.split('/')[-1]
    if 'offers' in name:
        args = dict(ylim=[0., .6])
        xlabel = 'Last turn'
        den = 'threads'
    elif 'threads' in name:
        args = dict(ylim=[0, 1])
        xlabel = 'Number of threads'
        den = 'listings'
    elif MSG in name:
        args = dict(xlim=[1, 6],
                    ylim=[0, .1],
                    xticks=range(1, 7),
                    ds='default')
        xlabel = 'Turn'
        den = 'eligible offers'
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    bar_plot(df, **args)
    save_fig(path, xlabel=xlabel,
             ylabel='Fraction of {}'.format(den),
             legend=isinstance(df, pd.DataFrame))


def area_plot(df, xlim=None, ylim=None, xticks=None, yticks=None):
    plt.clf()
    df.plot.area(xlim=xlim, ylim=ylim, cmap=plt.get_cmap('plasma'))

    # ticks
    if xticks is not None:
        plt.gca().set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)


def action_plot(path, df, turn=None):
    area_plot(df, ylim=[0, 1])
    save_fig(path, reverse_legend=True,
             xlabel='{} offer as fraction of list price'.format(
                 'Seller' if turn in IDX[BYR] else 'Buyer'))
