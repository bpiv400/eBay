import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from constants import FIG_DIR, IDX, BYR, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, \
    DAY, HOUR, EPS
from featnames import CON, NORM, DELAY, ARRIVAL, MSG, ACCEPT, \
    REJECT, META, CNDTN

FONTSIZE = {'training': 24}  # fontsize by plot type

plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = True


def get_name(path):
    return path.split('/')[-1].split('_')[1]


def remove_leading_zero(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if 0 < np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def save_fig(path, legend=True, legend_kwargs=None, reverse_legend=False,
             xlabel=None, ylabel=None, square=True, xaxis=True, yaxis=True,
             xticks=None, yticks=None, xticklabels=None, yticklabels=None,
             xlim=None, ylim=None, gridlines=True, integer_xticks=False, logx=False):
    name = path.split('/')[-1]
    cat = name.split('_')[0]

    # font size
    fontsize = 16 if cat not in FONTSIZE else FONTSIZE[cat]

    # axis limits
    if xlim is not None:
        plt.gca().set_xbound(xlim)
    if ylim is not None:
        plt.gca().set_ybound(ylim)

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


def add_diagonal(df):
    low, high = df.index.min(), df.index.max()
    plt.plot([low, high], [low, high], '-k', lw=0.5)


def add_vline(x=None, y=None):
    plt.plot([x, x], y, '-k', lw=1)


def cdf_plot(path, df):
    name = get_name(path)

    # labels and plot arguments
    den, ylim, vline = 'listings', [0, 1], None
    if name == 'price':
        df.index = np.log10(df.index)
        ticks = [1, 10, 100, 1000]
        args = dict(xlim=[0, 3],
                    xticks=np.log10(ticks),
                    xticklabels=['${}'.format(i) for i in ticks],
                    xlabel='Sale price')
        den = 'sales'
    elif name == NORM:
        args = dict(xlim=[0, 1],
                    xlabel='Sale price / list price')
        den = 'sales'
    elif name.startswith('days'):
        args = dict(xlim=[0, 1],
                    xlabel='Fraction of listing window')
    elif name.startswith(ARRIVAL):
        upper = MAX_DELAY_ARRIVAL / DAY
        args = dict(xticks=np.arange(0, upper, 3),
                    xlim=[0, upper], ylim=[0, 0.02],
                    xlabel='Days since listing start')
        den = 'buyers, by hour'
    elif name.startswith('hist'):
        args = dict(xlim=[0, 250],
                    xlabel='Prior best-offer threads for buyer')
        den = 'threads'
    elif name.startswith(DELAY):
        upper = int(MAX_DELAY_TURN / HOUR)
        args = dict(xlim=[0, upper],
                    xticks=np.arange(0, upper + EPS, 6),
                    xlabel='Response time in hours')
        den = 'offers'
    elif name == CON:
        args = dict(xlim=[0, 1],
                    # xticks=np.arange(0, 100 + 1e-8, 10),
                    xlabel='Concession')
        den = 'offers'
    elif name.startswith('values'):
        args = dict(xlim=[0, 1],
                    xlabel='value / list price')
    elif name.startswith('t1value'):
        vline = .5
        args = dict(xlim=[.1, .9],
                    xlabel='value / list price')
    elif name.startswith('netvalue'):
        vline = 0
        args = dict(xlim=[-.25, .25],
                    xlabel='(Value $-$ final seller offer) / list price')
    elif name == 'realval':
        args = dict(xlim=[0, 1],
                    xlabel='Realized value')
    elif name == 'contval':
        args = dict(xlim=[0, 1],
                    xlabel='Value of unsold items')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # create plot and save
    for c in df.columns:
        s = df[c].dropna()
        plt.plot(s.index, s, label=c, ds='steps-post')
    if vline is not None:
        add_vline(x=vline, y=ylim)
    save_fig(path,
             ylim=[0, 1],
             ylabel='Cumulative share of {}'.format(den),
             legend=isinstance(df, pd.DataFrame), **args)


def simple_plot(path, obj):
    name = get_name(path)
    if name == 'timevals':
        upper = np.ceil(obj.max() * 5) / 5
        lower = np.floor(obj.min() * 5) / 5
        args = dict(xlim=[0, 1], ylim=[lower, upper],
                    xlabel='Fraction of listing window elapsed',
                    ylabel='Average normalized value of unsold items')
    elif name == 'slrvals':
        args = dict(logx=True,
                    xlabel='Seller reviews',
                    ylabel='Average normalized value')
    elif name == 'slrsale':
        args = dict(logx=True,
                    xlabel='Seller reviews',
                    ylabel='Sale rate')
    elif name == 'slrnorm':
        args = dict(logx=True,
                    xlabel='Seller reviews',
                    ylabel='Average normalized sale price')
    elif name == 'valcon':
        args = dict(ylim=[0, .5], xlabel='Value',
                    ylabel='Average buyer concession')
    elif name == 'rejacc':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='First buyer offer / list price',
                    ylabel='Pr(accept) in turn 3')
    elif name == 'rejrej':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='First buyer offer / list price',
                    ylabel='Pr(reject) in turn 3')
    elif name == 'rejnorm':
        args = dict(xlim=[.4, 1], ylim=[.4, 1],
                    xlabel='First buyer offer / list price',
                    ylabel='Turn 3 buyer offer / list price')
    elif name == 'roc':
        args = dict(xlim=[0, 1], ylim=[0, 1],
                    xlabel='False positive rate',
                    ylabel='True positive rate')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    plt.clf()

    if type(obj) is pd.Series:
        plt.plot(obj.index, obj, '-k')
        if name == 'roc':
            add_diagonal(obj)

        save_fig(path, legend=False, **args)

    elif len(obj.columns.names) == 1:
        plot_ci = list(obj.columns) == ['beta', 'err']

        if plot_ci:
            plt.plot(obj.index, obj.beta, '-k')
            plt.plot(obj.index, obj.beta + obj.err, '--k')
            plt.plot(obj, obj.beta - obj.err, '--k')
        else:
            for c in obj.columns:
                plt.plot(obj.index, obj[c], label=c)

        save_fig(path, legend=(not plot_ci), **args)

    else:
        colors = ['r', 'g', 'b']
        dsets = obj.columns.get_level_values(0).unique()
        for i in range(len(dsets)):
            dset = dsets[i]
            df_i = obj.xs(dset, level=0, axis=1)
            plt.plot(df_i.index, df_i.beta, '-', color=colors[i], label=dset)
            plt.plot(df_i.index, df_i.beta + df_i.err, '--', color=colors[i])
            plt.plot(df_i.index, df_i.beta - df_i.err, '--', color=colors[i])

        if name == 'rejnorm':
            add_diagonal(obj)

        save_fig(path, legend=True, **args)


def training_plot(path, df):
    plt.clf()

    plt.plot(df.index, df.test, '-k')
    plt.plot(df.index, df.train, '--k')
    plt.plot(df.index, df.baserate, '-k')

    save_fig(path,
             integer_xticks=True,
             legend=False,
             xlabel='Epoch',
             ylabel='')


def draw_response(line=None, dots=None, label=None, c='k'):
    plt.plot(line.index, line.beta, '-', label=label, color=c)
    if 'err' in line.columns:
        plt.plot(line.index, line.beta + line.err, '--', color=c)
        plt.plot(line.index, line.beta - line.err, '--', color=c)

    if dots is not None:
        for i in range(len(dots.index)):
            x = dots.index[i]
            row = dots.iloc[i]
            plt.plot(x, row['beta'], 'o', color=c, ms=3)
            if 'err' in dots.columns:
                high = row['beta'] + row['err']
                low = row['beta'] - row['err']
                plt.plot([x, x], [low, high], '-', color=c)


def response_plot(path, obj):
    name = get_name(path)

    if name.startswith('bin'):
        ticks = [10, 20, 50, 100, 250, 1000]
        args = dict(xticks=np.log10(ticks),
                    xticklabels=['${}'.format(t) for t in ticks],
                    xlabel='List price')

        if name == 'bin':
            args['ylim'] = [.7, 1]
            args['ylabel'] = 'Average response to 50% first offer'
        elif name == 'binvals':
            args['ylabel'] = 'Average normalized value'
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))

    elif name in [ACCEPT, REJECT, CON]:
        ylabel = 'Average seller concession' if name == CON \
            else 'Pr(seller {}s)'.format(name)
        args = dict(xlim=[.4, .99], ylim=[0, 1],
                    xlabel='First buyer offer / list price',
                    ylabel=ylabel)
    elif name == NORM:
        args = dict(xlim=[.4, 1], ylim=[.4, 1],
                    xlabel='First buyer offer / list price',
                    ylabel='Avg seller counter / list price')
    elif name == 'hist':
        args = dict(xlim=[.4, 1], ylim=[0, .7],
                    xlabel='First buyer offer / list price',
                    ylabel='Average buyer experience percentile')
    elif name == 'rejnorm':
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Average buyer concession in turn 3')
    elif name == 'accnorm':
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Pr(accept) in turn 3')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is tuple:
        line, dots = obj
    else:
        line, dots = obj, None

    if len(line.columns.names) > 1:
        # first plot data by itself
        draw_response(line=line.xs('Data', level=0, axis=1),
                      dots=dots.xs('Data', level=0, axis=1))
        if name == NORM:
            add_diagonal(line)
        save_fig(path, legend=False, **args)

        # then plot data with one agent run
        dsets = line.columns.get_level_values(0).unique()
        agent_runs = [dset for dset in dsets if dset.startswith('Agent')]
        for i in range(len(agent_runs)):
            agent_run = agent_runs[i]
            delta = agent_run.split('_'[-1])
            draw_response(line=line.xs('Data', level=0, axis=1),
                          dots=dots.xs('Data', level=0, axis=1),
                          label='Data', c='r')
            draw_response(line=line.xs(agent_run, level=0, axis=1),
                          dots=dots.xs(agent_run, level=0, axis=1),
                          label='Agent', c='b')
            if name == NORM:
                add_diagonal(line)
            save_fig('{}_{}'.format(path, delta), legend=True, **args)

    else:
        draw_response(line=line, dots=dots)
        if name == NORM:
            add_diagonal(line)
        save_fig(path, legend=False, **args)


def slr_plot(path, d):
    name = get_name(path)
    if name == 'vals':
        save_args = dict(ylabel='Average normalized value')
    elif name == 'sale':
        save_args = dict(ylabel='Pr(sale)')
    elif name == 'norm':
        save_args = dict(ylabel='Average sale price, among sales')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    colors = {'Store': 'r', 'No store': 'b'}
    for k in ['Store', 'No store']:
        df = d[k]
        x = df.index
        c = colors[k]
        plt.plot(x, df.beta, color=c, ls='-', label=k)
        plt.plot(x, df.beta + df.err, color=c, ls='--')
        plt.plot(x, df.beta - df.err, color=c, ls='--')

    xticks = np.arange(2, 5)
    xticklabels = (10 ** xticks).astype(np.int64)
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xticklabels)

    save_fig(path, xlabel='Seller reviews', **save_args)


def coef_plot(path, df):
    name = get_name(path)
    if name == 'photovals':
        args = dict(xlim=[.45, .61],
                    ylabel='Number of photos',
                    xlabel='Average normalized value')
    elif name == 'dowvals':
        args = dict(xlim=[.46, .57], ylabel='',
                    xlabel='Average normalized value')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    plt.barh(df.index, df.beta, xerr=df.err, color='none')
    plt.scatter(df.beta, df.index, s=20, color='black')
    plt.gca().invert_yaxis()  # labels read top-to-bottom

    save_fig(path,
             legend=False,
             yticks=range(len(df.index)),
             yticklabels=df.index,
             **args)


def bar_plot(path, df):
    name = get_name(path)
    if name == 'offers':
        args = dict(ylim=[0., .5],
                    xlabel='Turn of last offer',
                    ylabel='Fraction of threads')
    elif name == 'threads':
        args = dict(ylim=[0, 1],
                    xlabel='Number of threads',
                    ylabel='Fraction of listings')
    elif name == MSG:
        args = dict(ylim=[0, .5],
                    xlabel='Turn',
                    ylabel='Fraction of eligible offers')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    df.plot.bar(rot=0)

    save_fig(path, xticklabels=df.index, **args)


def draw_area(df, xlim=None, ylim=None):
    df.plot.area(xlim=xlim, ylim=ylim, cmap=plt.get_cmap('plasma'))


def action_plot(path, df, turn=None):
    draw_area(df, ylim=[0, 1])
    save_fig(path, reverse_legend=True,
             xlabel='{} offer as fraction of list price'.format(
                 'Seller' if turn in IDX[BYR] else 'Buyer'))


def draw_contour(s, vmin=None, vmax=None):
    idx = [s.index.get_level_values(i) for i in range(2)]
    X = np.unique(idx[0])
    Y = np.unique(idx[1])
    Z = np.reshape(s.values, (len(Y), len(X)))

    lower = np.floor(s.min() * 100) / 100
    upper = np.ceil(s.max() * 100) / 100
    levels = np.arange(lower, upper, .002)
    subset = [i for i in levels if int(i * 1000) % 5 == 0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels,
                    cmap=plt.get_cmap('plasma'),
                    vmin=vmin, vmax=vmax)
    ax.clabel(CS, inline=True, fontsize=10, levels=subset)


def contour_plot(path, s):
    name = get_name(path)
    if name == 'normval':
        inc = .002
        args = dict(xlabel='Seller counter / list price',
                    ylabel='Value')
    elif name == 'delayacc':
        inc = .01
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller concession')
    elif name.startswith('hist'):
        inc = .01
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller experience percentile')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    idx = [s.index.get_level_values(i) for i in range(2)]
    X = np.unique(idx[0])
    Y = np.unique(idx[1])
    Z = np.reshape(s.values, (len(Y), len(X)))

    lower = np.floor(s.min() * 100) / 100
    upper = np.ceil(s.max() * 100) / 100
    levels = np.arange(0, 1 + inc, inc)
    subset = [levels[i] for i in range(len(levels)) if i % 5 == 0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels,
                    cmap=plt.get_cmap('plasma'),
                    vmin=lower, vmax=upper)
    ax.clabel(CS, inline=True, fontsize=10, levels=subset)

    save_fig(path, legend=False, **args)


def draw_scatter(df, cmap=None, **plot_args):
    plt.scatter(df.x, df.y, s=(df.s / 1e3), c=df.c,
                cmap=plt.get_cmap(cmap), **plot_args)

    if cmap == 'plasma':
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=6)


def add_labels(labels):
    for label in labels.index:
        plt.text(labels.loc[label, 'x'], labels.loc[label, 'y'], label,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=6)


def w2v_plot(path, df):
    name = path.split('/')[-1].split('_')[1]
    if name == META:
        plot_args = dict(cmap='prism')
    elif name in ['values', 'sale']:
        plot_args = dict(cmap='plasma', vmin=0, vmax=1)
    elif name in [CNDTN, 'norm', 'sale']:
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


def pdf_plot(path, obj):
    name = get_name(path)
    if name == 'values':
        args = dict(xlim=[0, 1], xlabel='Value')
    elif name == 'arrival':
        args = dict(xlim=[0, 1],
                    xlabel='Fraction of listing window')
    elif name == 'interarrival':
        args = dict(xlim=[0, 36], xlabel='Hours',
                    xticks=np.arange(0, 36 + EPS, 6))
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    legend = type(obj) is pd.DataFrame
    if not legend:
        upper = obj.max()
        plt.plot(obj.index, obj)
    else:
        upper = obj.max().max()
        for c in obj.columns:
            plt.plot(obj.index, obj[c], label=c)

    save_fig(path,
             legend=legend,
             yaxis=False,
             ylim=[0, upper],
             gridlines=False,
             **args)
