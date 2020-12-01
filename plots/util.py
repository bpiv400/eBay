import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from viscid.plot import vpyplot as vlt
from plots.const import BIN_TICKS, SLRBO_TICKS
from constants import FIG_DIR, IDX, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, \
    DAY, HOUR, EPS
from featnames import CON, NORM, DELAY, ARRIVAL, MSG, ACCEPT, \
    REJECT, META, CNDTN, BYR, SLR, EXP

FONTSIZE = {'training': 24}  # fontsize by plot type

plt.style.use('seaborn-colorblind')
COLORS = ['k'] + vlt.get_current_colorcycle()

mpl.rcParams['axes.grid'] = True


def get_name(path):
    return path.split('/')[-1].split('_')[1]


def get_log_ticks(idx=None, ticks=BIN_TICKS):
    upper = 10 ** idx.max()
    lower = 10 ** idx.min()
    ticks = [t for t in ticks if lower <= t <= upper]
    return ticks


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


def save_fig(path, legend=True, legend_kwargs=None, reverse_legend=False,
             xlabel=None, ylabel=None, square=True, xaxis=True, yaxis=True,
             xticks=None, yticks=None, xticklabels=None, yticklabels=None,
             xlim=None, ylim=None, gridlines=True, integer_xticks=False, logx=False):
    name = get_name(path)

    # font size
    fontsize = 16 if name not in FONTSIZE else FONTSIZE[name]

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
                   handlelength=1.,
                   fancybox=False,
                   frameon=gridlines,
                   fontsize=fontsize,
                   title_fontsize=fontsize)
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


def cdf_plot(path, obj):
    name = get_name(path)

    agent = path.startswith(BYR) or path.startswith(SLR)
    den = 'valid listings' if agent else 'listings'

    # labels and plot arguments
    ylim, vline = [0, 1], None
    if name in ['lstgprice', 'saleprice']:
        obj.index = np.log10(obj.index)
        ticks = range(4)
        args = dict(xlim=[0, 3],
                    xticks=ticks,
                    xticklabels=['$10^{}$'.format(i) for i in ticks],
                    xlabel='Sale price ($)')
        if name == 'saleprice':
            den = 'sales'
    elif name in ['lstgnorm', 'salenorm']:
        args = dict(xlim=[0, 1],
                    xlabel='Sale price / list price')
        if name == 'salenorm':
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
        args = dict(xlim=[0, 1], xlabel='Concession')
        den = 'offers'
    elif name == NORM:
        args = dict(xlim=[0, 1], xlabel='Offer / list price')
        den = 'offers'
    elif name in ['values', 'unsoldvals', 'soldvals']:
        args = dict(xlim=[0, 1],
                    xlabel='Value / list price')
    elif name.startswith('t1value'):
        vline = .5
        args = dict(xlim=[.1, .9],
                    xlabel='Value / list price')
    elif name.startswith('t7value'):
        vline = 0
        args = dict(xlim=[-.25, .25],
                    xlabel='(Value $-$ final seller offer) / list price')
    elif name == 'realval':
        args = dict(xlim=[0, 1],
                    xlabel='Realized value')
    elif name == 'bin':
        ticks = [10] + list(range(200, 1001, 200))
        args = dict(xlim=[9.95, 1000], xlabel='List price',
                    xticks=ticks,
                    xticklabels=['${}'.format(t) for t in ticks])
    elif name == 'autoacc':
        args = dict(xlim=[0, 1], xlabel='Accept price / list price')
    elif name == 'autodec':
        args = dict(xlim=[0, 1], xlabel='Decline price / list price')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # create plot and save
    if type(obj) is pd.Series:
        plt.plot(obj.index, obj, ds='steps-post', color='k')
    else:
        if 'Data' in obj.columns:
            assert obj.columns[0] == 'Data'
            s = obj['Data'].dropna()

            # plot separately
            plt.plot(s.index, s, ds='steps-post', color='k')
            if vline is not None:
                add_vline(x=vline, y=ylim)
            save_fig('{}_Data'.format(path), ylim=[0, 1],
                     ylabel='Cumulative share of {}'.format(den),
                     legend=False, **args)

            # plot data and each agent
            if len(obj.columns) > 2:
                for i in range(1, len(obj.columns)):
                    plt.plot(s.index, s, ds='steps-post', color='k',
                             label='Data' if 'Simulations' in obj.columns else 'Humans')
                    s_agent = obj.iloc[:, i].dropna()
                    label = obj.columns[i]
                    plt.plot(s_agent.index, s_agent, ds='steps-post',
                             color=COLORS[i], label=label)
                    save_fig('{}_{}'.format(path, label[10:13]),
                             ylim=[0, 1],
                             ylabel='Cumulative share of {}'.format(den),
                             legend=True, **args)

            # plot together
            plt.plot(s.index, s,
                     label='Data' if 'Simulations' in obj.columns else 'Humans',
                     ds='steps-post', color='k')
            df = obj.drop('Data', axis=1)
        else:
            df = obj
        for c in df.columns:
            s = df[c].dropna()
            plt.plot(s.index, s, label=c, ds='steps-post')

    if vline is not None:
        add_vline(x=vline, y=ylim)
    save_fig(path, ylim=[0, 1],
             ylabel='Cumulative share of {}'.format(den),
             legend=True, **args)


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
    elif name == 'roc':
        args = dict(xlim=[0, 1], ylim=[0, 1],
                    xlabel='False positive rate',
                    ylabel='True positive rate')
    elif name == 'slrrejaccbin':
        ticks = get_log_ticks(obj.index)
        args = dict(ylim=[0, 1],
                    xticks=np.log10(ticks),
                    xticklabels=ticks,
                    xlabel='List price ($)',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'interarrival':
        args = dict(ylim=[0, 7], xlim=[0, 2],
                    xticks=[0, .5, 1, 1.5, 2],
                    xlabel='Days to first arrival',
                    ylabel='Days between first two arrivals',
                    legend_kwargs=dict(title='First offer'))
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
            plt.plot(obj.index, obj.beta - obj.err, '--k')
        else:
            for c in obj.columns:
                plt.plot(obj.index, obj[c], label=c)

        save_fig(path, legend=(not plot_ci), **args)

    else:
        dsets = obj.columns.get_level_values(0).unique()
        for i in range(len(dsets)):
            dset = dsets[i]
            df_i = obj.xs(dset, level=0, axis=1)
            plt.plot(df_i.index, df_i.beta, '-', color=COLORS[i+1], label=dset)
            plt.plot(df_i.index, df_i.beta + 1.96 * df_i.err, '--', color=COLORS[i+1])
            plt.plot(df_i.index, df_i.beta - 1.96 * df_i.err, '--', color=COLORS[i+1])

        if NORM in name:
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


def draw_response(line=None, dots=None, diagonal=False, connect=False,
                  label=None, c='k'):
    if label is None or label == 'Data':
        plt.plot(line.index, line.beta, '-', color=c)
    else:
        plt.plot(line.index, line.beta, '-',
                 label=label, color=c)
    if 'err' in line.columns:
        plt.plot(line.index, line.beta + 1.96 * line.err, '--', color=c)
        plt.plot(line.index, line.beta - 1.96 * line.err, '--', color=c)

    if dots is not None:
        for i in range(len(dots.index)):
            x = dots.index[i]
            row = dots.iloc[i]
            plt.plot(x, row['beta'], 'o', color=c, ms=5)
            if 'err' in dots.columns:
                high = row['beta'] + 1.96 * row['err']
                low = row['beta'] - 1.96 * row['err']
                plt.plot([x, x], [low, high], '-', color=c)

    if diagonal:
        add_diagonal(line)

    if connect:
        y = dots.loc[0, 'beta']
        idx = (np.abs(line['beta'].values - y)).argmin()
        x = line.index[idx]
        plt.plot([0, x], [y, y], '-k', lw=1)
        plt.plot([x, x], [0, y], '-k', lw=1)


def response_plot(path, obj):
    name = get_name(path)

    if name.startswith('bin'):
        args = dict(xticks=np.log10(BIN_TICKS),
                    xticklabels=['${}'.format(t) for t in BIN_TICKS],
                    xlabel='List price')

        if name == 'bin':
            args['ylim'] = [.7, 1]
            args['ylabel'] = 'Average response to 50% first offer'
        elif name == 'binvals':
            args['ylabel'] = 'Normalized value'
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))

    elif name in [ACCEPT, REJECT, 'counter']:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Pr({})'.format(name))
    elif name == NORM:
        args = dict(xlim=[.4, 1], ylim=[.4, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Offer / list price')
    elif name == 'salenorm':
        args = dict(xlim=[.4, 1], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Sale price / list price')
    elif name == 'rewardnorm':
        args = dict(xlim=[.4, 1], ylim=[.4, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Reward / list price')
    elif name == 'hist':
        args = dict(xlim=[.4, 1], ylim=[0, .7],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Buyer experience percentile')
    elif name == 'accnorm':
        args = dict(xlabel='Turn 1: Concession',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'slrrejacc':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Concession',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'slrrejrej':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Concession',
                    ylabel='Turn 3: Pr(walk)')
    elif name == 'slrrejcon':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Concession',
                    ylabel='Turn 3: Concession')
    elif name == 'slrrejnorm':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Concession',
                    ylabel='Turn 3: Offer / list price')
    elif name == 'rejacc':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 3: Pr(accept)',
                    legend_kwargs=dict(title='Turn 2 reject type'))
    elif name == 'rejrej':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 3: Pr(walk)')
    elif name == 'rejnorm':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 3: Offer / list price')
    elif name == 'rejrejacc':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 3: Offer / list price',
                    ylabel='Turn 5: Pr(accept)')
    elif name == 'rejrejrej':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 3: Offer / list price',
                    ylabel='Turn 5: Pr(reject)')
    elif name == 'rejrejnorm':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 3: Offer / list price',
                    ylabel='Turn 5: Offer / list price')
    elif 'slrbo' in name:
        if name == 'expslrbo':
            ylabel = 'Expirations / manual rejects'
        elif name == 'slrbo':
            ylabel = 'Value / list price'
        elif name == 'slrbosale':
            ylabel = 'Pr(sale)'
        elif name == 'slrboprice':
            ylabel = 'Sale price'
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))
        args = dict(xlabel='Number of best offer listings',
                    ylabel=ylabel,
                    xticks=np.log10(SLRBO_TICKS),
                    xticklabels=SLRBO_TICKS)
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is tuple:
        line, dots = obj
    else:
        line, dots = obj, None

    if len(line.columns.names) > 1:
        dsets = line.columns.get_level_values(0).unique()

        # first plot data by itself
        if 'Data' in dsets:
            draw_response(line=line.xs('Data', level=0, axis=1),
                          dots=None if dots is None else dots.xs('Data', level=0, axis=1),
                          diagonal=(NORM in name),
                          connect=(name in ['slrrejrej', 'slrrejacc']))
            save_fig('{}_Data'.format(path), legend=False, **args)

        else:
            for dset in dsets:
                draw_response(line=line.xs(dset, level=0, axis=1),
                              dots=None if dots is None else dots.xs(dset, level=0, axis=1),
                              diagonal=(NORM in name),
                              connect=(name in ['slrrejrej', 'slrrejacc']),
                              label=dset)
                save_fig('{}_{}'.format(path, dset), legend=False, **args)

        # then plot data all together
        if len(dsets) < len(COLORS):
            for i in range(len(dsets)):
                dset = dsets[i]
                draw_response(line=line.xs(dset, level=0, axis=1),
                              dots=None if dots is None else dots.xs(dset, level=0, axis=1),
                              diagonal=(NORM in name),
                              label=dset, c=COLORS[i])

            save_fig(path, legend=True, **args)

    else:
        draw_response(line=line, dots=dots, diagonal=(NORM in name))
        save_fig(path, legend=False, **args)


def coef_plot(path, df):
    name = get_name(path)
    if name == 'photovals':
        args = dict(xlim=[.31, .45],
                    ylabel='Number of photos',
                    xlabel='Normalized value')
    elif name == 'dowvals':
        args = dict(xlim=[.34, .38], ylabel='',
                    xlabel='Normalized value')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    plt.barh(df.index, df.beta, xerr=(1.96 * df.err), color='none')
    plt.scatter(df.beta, df.index, s=20, color='black')
    plt.gca().invert_yaxis()  # labels read top-to-bottom

    save_fig(path,
             legend=False,
             yticks=range(len(df.index)),
             yticklabels=df.index,
             **args)


def bar_plot(path, obj):
    name = get_name(path)
    if name == 'offers':
        args = dict(ylim=[0., .6],
                    xlabel='Turn of last offer',
                    ylabel='Fraction of threads')
    elif name == 'threads':
        agent = path.startswith(BYR) or path.startswith(SLR)
        ylabel = 'valid listings' if agent else 'listings'
        args = dict(ylim=[0, 1],
                    xlabel='Number of threads',
                    ylabel='Fraction of ' + ylabel)
    elif name == MSG:
        args = dict(ylim=[0, .5],
                    xlabel='Turn',
                    ylabel='Fraction of eligible offers')
    elif name == 'norm':
        lower = np.floor(obj.min() * 100) / 100 - .01
        upper = np.ceil(obj.max() * 100) / 100 + .01
        args = dict(ylim=[lower, upper],
                    legend=False, xlabel='',
                    ylabel='Reward / list price')
    elif name == 'dollar':
        lower, upper = np.floor(obj.min()) - 1, np.ceil(obj.max()) + 1
        args = dict(ylim=[lower, upper],
                    legend=False, xlabel='',
                    ylabel='Reward ($)')
    elif name == 'training':
        baserate = obj['Baserate']
        obj.drop('Baserate', inplace=True)
        args = dict(ylim=[baserate, None], legend=False,
                    xlabel='', ylabel='')
    elif name == REJECT:
        args = dict(xlabel='Turn', ylabel='Pr(reject)')
    elif name == EXP:
        args = dict(xlabel='Turn', ylabel='Pr(expire)')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is pd.Series:
        if 'Data' in obj.index:
            obj.rename({'Data': 'Humans'}, inplace=True)
        obj.plot.bar(rot=0)
    else:
        rot = 45 if len(obj.columns) > 3 else 0
        obj.plot.bar(rot=rot, color=COLORS)
    save_fig(path, xticklabels=obj.index, gridlines=False, **args)


def draw_area(df, xlim=None, ylim=None):
    df.plot.area(xlim=xlim, ylim=ylim, cmap=plt.get_cmap('plasma'))


def action_plot(path, df, turn=None):
    draw_area(df, ylim=[0, 1])
    save_fig(path, reverse_legend=True,
             xlabel='{} offer as fraction of list price'.format(
                 'Seller' if turn in IDX[BYR] else 'Buyer'))


def draw_contour(s=None, inc=.01):
    idx = [s.index.get_level_values(i) for i in range(2)]
    X = np.unique(idx[0])
    Y = np.unique(idx[1])
    Z = np.reshape(s.values, (len(Y), len(X)))

    vmin = np.floor(s.min() * 100) / 100
    vmax = np.ceil(s.max() * 100) / 100
    levels = np.arange(0, np.ceil(vmax) + inc, inc)
    subset = [levels[i] for i in range(len(levels)) if i % 5 == 0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels,
                    cmap=plt.get_cmap('plasma'),
                    vmin=vmin, vmax=vmax)
    ax.clabel(CS, inline=True, fontsize=14, levels=subset)


def contour_plot(path, s):
    name, inc = get_name(path), .01
    if name == 'normval':
        inc = .002
        args = dict(xlabel='Seller counter / list price',
                    ylabel='Value')
    elif name == 'delayacc':
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller concession')
    elif name.startswith('hist'):
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller experience percentile')
    elif 'rejbin' in name:
        ticks = get_log_ticks(s.index.levels[1])
        args = dict(yticks=np.log10(ticks),
                    yticklabels=ticks,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='List price ($)')
    elif name == 'rejdays':
        args = dict(ylabel='Days to first offer',
                    xlabel='Turn 1: Offer / list price')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    draw_contour(s, inc=inc)
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
    if name == 'arrival':
        args = dict(xlim=[0, 1],
                    xlabel='Fraction of listing window')
    elif name == 'interarrival':
        args = dict(xlim=[0, 48], xlabel='Hours',
                    xticks=np.arange(0, 48 + EPS, 12))
    elif name == 'values':
        args = dict(xlim=[0, 1],
                    xlabel='Value / list price')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    legend = type(obj) is pd.DataFrame
    if not legend:
        plt.plot(obj.index, obj, color='k')
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max()],
                 gridlines=False,
                 **args)

    elif 'Data' in obj.columns:
        s = obj['Data']

        # plot separately
        plt.plot(s.index, s, color='k')
        save_fig('{}_Data'.format(path),
                 legend=False,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)

        # plot together
        plt.plot(s.index, s, label='Data', color='k')
        df = obj.drop('Data', axis=1)
        for i in range(len(df.columns)):
            c = df.columns[i]
            plt.plot(df.index, df[c], label=c)
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)

    else:
        for i in range(len(obj.columns)):
            c = obj.columns[i]
            plt.plot(obj.index, obj[c], label=c)
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)
