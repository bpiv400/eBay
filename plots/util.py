import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from viscid.plot import vpyplot as vlt
from agent.const import TURN_COST_CHOICES
from plots.const import BIN_TICKS, SLRBO_TICKS, FONTSIZE
from constants import FIG_DIR, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, \
    DAY, HOUR, EPS
from featnames import CON, NORM, DELAY, ARRIVAL, MSG, ACCEPT, \
    REJECT, META, CNDTN, BYR, SLR, EXP

plt.style.use('seaborn-colorblind')
COLORS = ['k'] + vlt.get_current_colorcycle()
TRICOLOR = {'Humans': COLORS[0],
            'Agent': COLORS[1],
            'Impatient agent': COLORS[1],
            'Patient agent': COLORS[2]}

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
             xlim=None, ylim=None, gridlines=True, integer_xticks=False,
             logx=False, logy=False, legend_outside=False, fontsize=FONTSIZE, **args):

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


def add_diagonal(df):
    low, high = df.index.min(), df.index.max()
    plt.plot([low, high], [low, high], '-k', lw=0.5)


def add_line(x=None, y=None):
    if type(x) is list:
        plt.plot(x, [y, y], '-k', lw=1)
    elif type(y) is list:
        plt.plot([x, x], y, '-k', lw=1)
    else:
        raise ValueError('x or y must a list.')


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
    elif name == 't1value':
        vline = .5
        args = dict(xlim=[.1, .9],
                    xlabel='Value / list price')
    elif name in ['t3value', 't5value']:
        vline = 0
        args = dict(xlim=[-.25, .25],
                    xlabel='(Value $-$ smallest counter) / list price')
    elif name == 't7value':
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
    elif name == 'discount':
        args = dict(xlim=[0, .5], xlabel='Discount / list price')
    elif name == 'totaldiscount':
        args = dict(xlim=[0, 200], xlabel='Discount ($)')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # create plot and save
    if type(obj) is pd.Series:
        plt.plot(obj.index, obj, ds='steps-post', color='k')
    else:
        if 'Humans' in obj.columns or 'Data' in obj.columns:
            assert obj.columns[0] in ['Humans', 'Data']
            c0 = obj.columns[0]
            s = obj[c0].dropna()

            # plot separately
            plt.plot(s.index, s, ds='steps-post', color='k')
            if vline is not None:
                add_line(x=vline, y=ylim)
            save_fig('{}_Data'.format(path), ylim=[0, 1],
                     ylabel='Cumulative share of {}'.format(den),
                     legend=False, **args)

            # plot data and each agent
            if len(obj.columns) > 2:
                for i in range(1, len(obj.columns)):
                    plt.plot(s.index, s, ds='steps-post', color='k', label=c0)
                    s_agent = obj.iloc[:, i].dropna()
                    label = obj.columns[i]
                    plt.plot(s_agent.index, s_agent, ds='steps-post',
                             color=COLORS[i], label=label)
                    save_fig('{}_{}'.format(path, label[10:-1]),
                             ylim=[0, 1],
                             ylabel='Cumulative share of {}'.format(den),
                             legend=True, **args)

            # plot together
            plt.plot(s.index, s, label=c0, ds='steps-post', color='k')
            df = obj.drop(c0, axis=1)
        else:
            df = obj
        for c in df.columns:
            s = df[c].dropna()
            plt.plot(s.index, s, label=c, ds='steps-post')

    if vline is not None:
        add_line(x=vline, y=ylim)
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
    elif name == 'hours':
        args = dict(xlim=[min(obj.index), max(obj.index)],
                    ylim=[0, 9],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Hours to response')
    elif name == 'norm2con3':
        args = dict(xlim=[.6, 1], ylim=[0, 1],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Concession')
    elif name == 'norm2acc3':
        args = dict(xlim=[.6, 1], ylim=[0, 1],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'norm2walk3':
        args = dict(xlim=[.6, 1], ylim=[0, 1],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr(walk)')
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
    elif name == 'rewardbin':
        args = dict(xlim=[1, 2.5],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Payoff / list price')
    elif name == 'discountbin':
        args = dict(xlim=[1, 2.5], ylim=[.1, .35],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount')
    elif name == 'offersbin':
        args = dict(xlim=[1, 2.5], ylim=[1, 2.4],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Number of buyer offers')
    elif name == 'avgdiscountbin':
        args = dict(xlim=[1, 2.5], ylim=[0, 50],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount / buyer offer ($)')
    elif name == 'discount':
        args = dict(xlim=[1, 2.5], ylim=[0, .4],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount / list price')
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
        colors = TRICOLOR if dsets[0] == 'Humans' else COLORS[1:]

        # plot together
        for i in range(len(dsets)):
            dset = dsets[i]
            df_i = obj.xs(dset, level=0, axis=1)
            c = colors[dset] if type(colors) is dict else colors[i]
            plt.plot(df_i.index, df_i.beta, '-', label=dset, color=c)
            if 'err' in df_i.columns:
                plt.plot(df_i.index, df_i.beta + 1.96 * df_i.err, '--',
                         color=c)
                plt.plot(df_i.index, df_i.beta - 1.96 * df_i.err, '--',
                         color=c)

        if name.endswith(NORM):
            add_diagonal(obj)

        save_fig(path, legend=True, **args)

        # plot separately
        for i in range(len(dsets)):
            dset = dsets[i]
            df_i = obj.xs(dset, level=0, axis=1)
            plt.plot(df_i.index, df_i.beta, '-k', label=dset)
            if 'err' in df_i.columns:
                plt.plot(df_i.index, df_i.beta + 1.96 * df_i.err, '--k')
                plt.plot(df_i.index, df_i.beta - 1.96 * df_i.err, '--k')

            if name.endswith(NORM):
                add_diagonal(obj)

            save_fig('{}_{}'.format(path, dset), legend=False, **args)


def training_plot(path, df):
    plt.clf()

    plt.plot(df.index, df.test, '-k')
    plt.plot(df.index, df.train, '--k')
    plt.plot(df.index, df.baserate, '-k')

    save_fig(path,
             integer_xticks=True,
             legend=False,
             xlabel='Epoch',
             ylabel='',
             fontsize=24)


def draw_response(line=None, dots=None, diagonal=False, connect=False,
                  label=None, c='k'):
    if label is None:
        plt.plot(line.index, line.beta, '-', color=c)
    else:
        plt.plot(line.index, line.beta, '-', label=label, color=c)
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
        y = dots.loc[1, 'beta']
        idx = (np.abs(line['beta'].values - y)).argmin()
        x = line.index[idx]
        plt.plot([x, 1], [y, y], '-k', lw=1)
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

    elif name in [ACCEPT, REJECT]:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Pr({})'.format(name))
    elif name == CON:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Concession')
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
                    ylabel='Payoff / list price')
    elif name == 'hist':
        args = dict(xlim=[.4, 1.01], ylim=[0, .7],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Buyer experience percentile')
    elif name == 'accnorm':
        args = dict(xlabel='Turn 1: Concession',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'slrrejacc':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr(accept)')
    elif name == 'slrrejrej':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr(walk)')
    elif name == 'slrrejcon':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Concession')
    elif name == 'slrrejnorm':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Offer / list price',
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
    elif name == 'norm2con3':
        args = dict(xlim=[.4, 1], ylim=[0, 1],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Concession')
    elif name in ['offer2walk', 'offer2acc']:
        ylabel = 'walk' if name.endswith('walk') else 'accept'
        args = dict(xlim=[.65, 1], ylim=[0, 1],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr({})'.format(ylabel))
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
        if 'Humans' in dsets:
            draw_response(line=line.xs('Humans', level=0, axis=1),
                          dots=None if dots is None else dots.xs('Humans', level=0, axis=1),
                          diagonal=name.endswith(NORM))
            save_fig('{}_Data'.format(path), legend=False, **args)

        else:
            for dset in dsets:
                draw_response(line=line.xs(dset, level=0, axis=1),
                              dots=None if dots is None else dots.xs(dset, level=0, axis=1),
                              diagonal=name.endswith(NORM),
                              label=dset)
                save_fig('{}_{}'.format(path, dset), legend=False, **args)

        # then plot data all together
        if len(dsets) < len(COLORS):
            for i in range(len(dsets)):
                dset = dsets[i]
                draw_response(line=line.xs(dset, level=0, axis=1),
                              dots=None if dots is None else dots.xs(dset, level=0, axis=1),
                              diagonal=name.endswith(NORM),
                              label=dset, c=COLORS[i])

            save_fig(path, legend=(name != REJECT), **args)

    else:
        draw_response(line=line, dots=dots,
                      diagonal=name.endswith(NORM))
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
    elif NORM in name:
        lower = np.floor(obj.min() * 100) / 100 - .01
        upper = np.ceil(obj.max() * 100) / 100 + .01
        ylim = [lower, upper]
        ylabel = '{} / list price'.format(
            'Discount' if name.endswith('2') else 'Payoff')
        args = dict(ylim=ylim,
                    legend=False, xlabel='',
                    ylabel=ylabel,
                    fontsize=20)
    elif 'dollar' in name:
        lower, upper = np.floor(obj.min()) - 1, np.ceil(obj.max()) + 1
        ylim = [lower, upper]
        ylabel = '{} ($)'.format('Discount' if name.endswith('2') else 'Payoff')
        args = dict(ylim=ylim,
                    legend=False, xlabel='',
                    ylabel=ylabel,
                    fontsize=20)
    elif name == 'training':
        baserate = obj['Baserate']
        obj.drop('Baserate', inplace=True)
        args = dict(ylim=[baserate, None], legend=False,
                    xlabel='', ylabel='')
    elif name == REJECT:
        args = dict(xlabel='Turn', ylabel='Pr(reject)')
    elif name == EXP:
        args = dict(xlabel='Turn', ylabel='Pr(expire)')
    elif name == 'saleturn':
        args = dict(xlabel='Turn', ylabel='Share of sales')
    elif name == 'discount':
        args = dict(xlabel='Turn', ylabel='Average discount ($)')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is pd.Series:
        obj.plot.bar(rot=0, color=COLORS[0])
        plt.gca().axhline(c='k', lw=1)
    else:
        rot = 45 if len(obj.columns) > 3 else 0
        obj.plot.bar(rot=rot)
    save_fig(path, xticklabels=obj.index, gridlines=False, **args)


def area_plot(path, df):
    name = get_name(path)
    if name == 'response':
        turn = path.split('_')[-1]
        args = dict(xlim=[min(df.index), max(df.index)],
                    ylim=[0, 1],
                    xlabel='Turn {}: Offer / list price'.format(turn),
                    ylabel='',
                    legend_outside=True)
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    df.plot.area(cmap=plt.get_cmap('plasma'))
    save_fig(path, reverse_legend=True, **args)


def draw_contour(s=None, vmin=0, vmax=1, inc=.01, zlabel=None,
                 reverse=False, **args):
    idx = [s.index.get_level_values(i) for i in range(2)]
    X = np.unique(idx[0])
    Y = np.unique(idx[1])
    Z = np.reshape(np.clip(s.values, vmin, vmax), (len(Y), len(X)))

    if reverse:
        cmap = plt.get_cmap('gnuplot_r')
    else:
        cmap = plt.get_cmap('gnuplot')
    levels = np.arange(0, vmax + inc, inc)
    plot_args = dict(levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    if vmax <= 1:
        subset = [levels[i] for i in range(len(levels)) if i % 10 == 0]
    else:
        subset = [level for level in levels if level % 1 == 0]

    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z, **plot_args)
    if zlabel is not None:
        cbar = fig.colorbar(CS, ticks=subset)
        cbar.ax.tick_params(labelsize=FONTSIZE)
        cbar.ax.set_ylabel(zlabel, fontsize=FONTSIZE)


def contour_plot(path, s):
    name = get_name(path)
    suffix = path.split('_')[-1]
    if name == 'normval':
        args = dict(xlabel='Seller counter / list price',
                    ylabel='Value',
                    inc=.002)
    elif name == 'delayacc':
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller concession')
    elif name.startswith('hist'):
        args = dict(xlabel='First buyer offer / list price',
                    ylabel='Seller experience percentile')
    elif name == 'interarrival':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival',
                    zlabel='Days between first two arrivals',
                    vmax=np.ceil(s.max()),
                    reverse=True)
    elif name == 'rejdays':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival',
                    zlabel='Turn 2: Pr(reject)' if suffix == 'store' else None)
    elif name == 'accdays':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival',
                    zlabel='Turn 2: Pr(accept)' if suffix == 'store' else None)
    elif 'rejbin' in name or name == 'normbin':
        if name == 'normbin':
            zlabel = 'Turn 2: Offer / list price'
        else:
            turn = 2 if name == 'rejbin' else 3
            action = 'reject' if turn == 2 else 'accept'
            zlabel = 'Turn {}: Pr({})'.format(turn, action)
        ticks = get_log_ticks(s.index.levels[1])
        args = dict(yticks=np.log10(ticks),
                    yticklabels=ticks,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='List price ($)',
                    zlabel=zlabel if suffix in ['data', name] else None)
    elif name == 'rejdays':
        args = dict(ylabel='Days to first offer',
                    xlabel='Turn 1: Offer / list price')
    elif name in ['offer2binwalk', 'offer2binacc']:
        turn = int(name[5])
        ticks = get_log_ticks(s.index.levels[1])
        zlabel = 'walk' if name.endswith('walk') else 'accept'
        args = dict(yticks=np.log10(ticks),
                    yticklabels=ticks,
                    xlabel='Turn {}: Offer / list price'.format(turn),
                    ylabel='List price ($)',
                    zlabel='Turn {}: Pr({})'.format(turn+1, zlabel),
                    vmax=.5)
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    draw_contour(s=s, **args)
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


def plot_humans(df=None, offsets=None):
    humans = df.loc[('Humans', None), :]
    plt.plot(humans.x, humans.y, 'Xk')
    plt.text(humans.x + offsets[0], humans.y + offsets[1], 'Humans')


def plot_agents(df=None, offsets=None, label=None, lspec='-o', color='k'):
    plt.plot(df.x, df.y, lspec, label=label, color=color)
    if offsets is not None:
        for label in df.index:
            plt.text(df.loc[label, 'x'] + offsets[0],
                     df.loc[label, 'y'] + offsets[1],
                     label)


def pareto_plot(path, df):
    name = get_name(path)
    if name == 'discount':
        args = dict(xlabel='Purchase rate',
                    ylabel='Discount on list price (%)',
                    ylim=[15, 40], xlim=[.5, 1])
        df['y'] *= 100
    elif name == 'dollar':
        args = dict(xlabel='Purchase rate',
                    ylabel='Discount on list price ($)',
                    ylim=[15, 40], xlim=[.5, 1])
    elif name == 'sales':
        args = dict(xlabel='Purchase rate',
                    ylabel='Savings on observed sale price ($)',
                    ylim=[0, 20], xlim=[.5, 1])
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    offsets = [(args[k][1] - args[k][0]) / 100
               for k in ['xlim', 'ylim']]

    # zero turn cost
    plot_humans(df=df, offsets=offsets)
    df0 = df.xs('0', level='turn_cost')
    plot_agents(df=df0, offsets=offsets)
    save_fig(path, legend=False, **args)

    # values item at list price
    names = {'$1-\\epsilon$': 'minus', '$1+\\epsilon$': 'plus'}
    for gamma in ['$1-\\epsilon$', '$1+\\epsilon$']:
        plot_humans(df=df, offsets=offsets)
        plot_agents(df=df0, lspec='-', color='gray')
        subset = df.xs(gamma, level='gamma')
        plot_agents(df=subset, offsets=offsets)
        save_fig('{}_{}'.format(path, names[gamma]), legend=False, **args)
