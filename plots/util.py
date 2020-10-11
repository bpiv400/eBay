import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from constants import FIG_DIR, IDX, BYR, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, \
    DAY, HOUR, MAX_DAYS
from featnames import CON, NORM, DELAY, ARRIVAL, BYR_HIST, MSG, ACCEPT, \
    REJECT, META, DELTA

FONTSIZE = {'training': 24}  # fontsize by plot type

plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = True


def save_fig(path, legend=True, legend_kwargs=None, reverse_legend=False,
             xlabel=None, ylabel=None, square=True, xaxis=True, yaxis=True):
    name = path.split('/')[-1]
    cat = name.split('_')[0]

    # font size
    fontsize = 16 if cat not in FONTSIZE else FONTSIZE[cat]

    if square:
        plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(),
                             adjustable='box')

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

    # axis labels and tick labels
    if xlabel is not None:
        plt.xticks(fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
    else:
        plt.xticks([])

    if ylabel is not None:
        plt.yticks(fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
    else:
        plt.yticks([])

    # hide axes
    if not xaxis or not yaxis:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
    if not xaxis:
        plt.gca().spines['bottom'].set_visible(False)
    if not yaxis:
        plt.gca().spines['left'].set_visible(False)

    # save
    plt.savefig(FIG_DIR + '{}.png'.format(path),
                format='png',
                # transparent=True,
                bbox_inches='tight')

    # close
    plt.close()


def draw_line(df, style=None, ds='steps-post', xlim=None, ylim=None,
              xticks=None, yticks=None, gridlines=True, diagonal=False,
              integer_xticks=False, logx=False, xticklabels=None):
    plt.clf()

    df.plot.line(style=style,
                 xlim=xlim,
                 ylim=ylim,
                 xticks=xticks,
                 yticks=yticks,
                 ds=ds,
                 legend=False,
                 logx=logx)

    if xticklabels is not None:
        plt.xticks(ticks=xticks, labels=xticklabels)

    if integer_xticks:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # gridlines
    if gridlines:
        plt.grid(axis='both',
                 which='both',
                 color='gray',
                 linestyle='-',
                 linewidth=0.5)
    else:
        plt.grid(axis='both', visible=False)

    # add 45 degree line
    if diagonal:
        low, high = df.index.min(), df.index.max()
        plt.gca().plot([low, high], [low, high], '-k', linewidth=0.5)


def add_beta_ci(df=None):
    for i in range(len(df.index)):
        x = df.index[i]
        plt.plot(x, df.beta.iloc[i], 'ok', ms=3)
        plt.plot([x, x], [df.low.iloc[i], df.high.iloc[i]], '-k')


def cdf_plot(path, df):
    names = path.split('/')[-1].split('_')
    dist, name = names[0], '_'.join(names[1:])

    # labels and plot arguments
    den, ylim = 'listings', [0, 1]
    if name.startswith('price'):
        df = df.loc[df.index > 0, :]
        args = dict(xlim=[1, 1000], logx=True)
        xlabel = 'Sale price'
    elif name.startswith(NORM):
        args = dict(xlim=[0, 1])
        xlabel = 'Sale price / list price'
    elif name.startswith('days'):
        args = dict(xlim=[0, MAX_DAYS])
        xlabel = 'Listing time in days'
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
        args = dict(xlim=[0, 100], xticks=np.arange(0, 100 + 1e-8, 10))
        xlabel = 'Concession (%)'
        den = 'offers'
    elif name.startswith('values'):
        args = dict(xlim=[0, 1])
        xlabel = 'Market value / list price'
    elif name.startswith('netvalue'):
        args = dict(xlim=[-1, 1])
        xlabel = '(Market value $-$ final seller offer) / list price'
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # create plot and save
    draw_line(df, ylim=[0, 1], **args)
    save_fig(path,
             xlabel=xlabel,
             ylabel='Cumulative share of {}'.format(den),
             legend=isinstance(df, pd.DataFrame))


def training_plot(path, df):
    draw_line(df,
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

    draw_line(df, diagonal=True, ds='default', **limits)
    save_fig(path, square=True, legend=isinstance(df, pd.DataFrame), **labels)


def response_plot(path, obj):
    if type(obj) is tuple:
        line, dots = obj
    else:
        line, dots = obj, None
    name = path.split('/')[-1].split('_')[-1].replace('.pkl', '')

    if name == 'bin':
        ticks = [10, 20, 50, 100, 250, 1000]
        line_args = dict(xlim=[1, 3], ylim=[.7, 1], logx=True,
                         xticks=np.log10(ticks),
                         xticklabels=['${}'.format(t) for t in ticks])
        save_args = dict(legend=True,
                         xlabel='List price',
                         ylabel='Avg response to 50% first offer')
    else:
        line_args = dict(xlim=[.4, .99])
        save_args = dict(legend=False,
                         xlabel='First buyer offer / list price')
        if name in [ACCEPT, REJECT]:
            line_args['ylim'] = [0, 1]
            save_args['ylabel'] = 'Pr(seller {}s)'.format(name)
        elif name in ['avg', CON]:
            line_args['ylim'] = [.4, .99]
            save_args['ylabel'] = 'Avg seller {} / list price'.format(
                'counter' if name == CON else 'response')
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))

    draw_line(line, style=['-k', '--k', '--k'], ds='default',
              diagonal=(name in [CON, 'avg']), **line_args)
    if dots is not None:
        add_beta_ci(df=dots)

    save_fig(path, **save_args)


def draw_bar(df, horizontal=False, stacked=False, rot=0,
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

    draw_bar(df, **args)
    save_fig(path, xlabel=xlabel,
             ylabel='Fraction of {}'.format(den),
             legend=isinstance(df, pd.DataFrame))


def draw_area(df, xlim=None, ylim=None, xticks=None, yticks=None):
    plt.clf()
    df.plot.area(xlim=xlim, ylim=ylim, cmap=plt.get_cmap('plasma'))

    # ticks
    if xticks is not None:
        plt.gca().set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)


def action_plot(path, df, turn=None):
    draw_area(df, ylim=[0, 1])
    save_fig(path, reverse_legend=True,
             xlabel='{} offer as fraction of list price'.format(
                 'Seller' if turn in IDX[BYR] else 'Buyer'))


def draw_contour(s, yticks=None, yticklabels=None):
    plt.clf()

    idx = [s.index.get_level_values(i) for i in range(2)]
    X = np.unique(idx[0])
    Y = np.unique(idx[1])
    Z = np.reshape(s.values, (len(Y), len(X)))

    lower = np.floor(s.min() * 100) / 100
    upper = np.ceil(s.max() * 100) / 100
    levels = np.arange(lower, upper, .01)
    subset = [i for i in levels if int(i * 100) % 5 == 0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels, cmap=plt.get_cmap('plasma'))
    ax.clabel(CS, inline=True, fontsize=10, levels=subset)

    # if yticks is not None:
    #     plt.yticks(ticks=yticks, labels=yticklabels)


def contour_plot(path, s):
    yticklabels = [10, 20, 50, 100, 200, 500, 1000]
    yticks = np.log10(yticklabels)
    draw_contour(s, yticks=yticks, yticklabels=yticklabels)
    save_fig(path, legend=False)


def draw_scatter(df, cmap=None, **plot_args):
    plt.clf()
    plt.scatter(df.x, df.y, s=(df.s / 1e4), c=df.c,
                cmap=plt.get_cmap(cmap), **plot_args)

    if cmap == 'plasma':
        plt.colorbar()


def add_labels(labels):
    for label in labels.index:
        plt.text(labels.loc[label, 'x'], labels.loc[label, 'y'], label,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=6)


def w2v_plot(path, df):
    name = path.split('/')[-1].split('_')[1]
    if name.startswith(META):
        plot_args = dict(cmap='prism')
    elif name.startswith(DELTA):
        plot_args = dict(cmap='plasma')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    draw_scatter(df, **plot_args)

    # meta labels
    labels = pd.DataFrame()
    den = df.groupby('label')['s'].sum()
    for var in ['x', 'y']:
        labels[var] = (df[var] * df['s']).groupby(df.label).sum() / den
    add_labels(labels)

    save_fig(path, xaxis=False, yaxis=False, legend=False)


def pdf_plot(path, df):
    draw_line(df, xlim=[0, 1], gridlines=False, ds='default')
    save_fig(path, xlabel='Market value / list price', yaxis=False)
