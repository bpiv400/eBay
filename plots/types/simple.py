import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from featnames import ACCEPT, REJECT, NORM, CON
from plots.const import BIN_TICKS, TRICOLOR, COLORS, SLRBO_TICKS
from plots.util import save_fig, get_name


def add_diagonal(df):
    low, high = df.index.min(), df.index.max()
    plt.plot([low, high], [low, high], '-k', lw=0.5)


def draw_simple(obj=None, dsets=None, i=None, colors=None):
    dset = dsets[i]
    df_i = obj.xs(dset, level=0, axis=1).dropna()
    c = colors[dset] if type(colors) is dict else colors[i]
    connect = '--' if dset.startswith('Heuristic') else '-'
    label = None if dset == 'Humans' and '$1' in dsets else dset
    plt.plot(df_i.index, df_i.beta, connect, color=c, label=label)
    if 'err' in df_i.columns:
        plt.plot(df_i.index, df_i.beta + 1.96 * df_i.err, '--', color=c)
        plt.plot(df_i.index, df_i.beta - 1.96 * df_i.err, '--', color=c)


def simple_plot(path, obj):
    name = get_name(path)
    diagonal = False
    if name == 'roc':
        args = dict(xlim=[0, 1], ylim=[0, 1],
                    xlabel='False positive rate',
                    ylabel='True positive rate')
        diagonal = True
    elif name in [ACCEPT, REJECT]:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Pr({})'.format(name),
                    legend=(name == ACCEPT))
    elif name == CON:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Concession')
    elif name == 'rejacc':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 3: Pr(accept)',
                    legend_kwargs=dict(title='Turn 2 reject type'))
    elif name == 'rejrej':
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 3: Pr(walk)')
    elif name == 'rewardbin':
        delta = float(path.split('_')[-1])
        ylim = [.54, .71] if delta == 0 else [.64, .77]
        loc = 'lower left' if delta == 0 else 'lower right'
        args = dict(xlim=[min(obj.index), max(obj.index)],
                    ylim=ylim,
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Payoff / list price',
                    legend_kwargs=dict(loc=loc))
    elif name == 'listbin':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[0, 1],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Turn 1: Pr(accept)',
                    legend_kwargs=dict(title='Offer cost'))
    elif name == 'listoffers':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[1, 2.5],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    yticks=np.arange(1, 3, .5),
                    ylabel='Buyer offers per thread',
                    legend=False)
    elif 'slrbo' in name:
        if name == 'slrbo':
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

    if type(obj) is pd.Series:
        plt.plot(obj.index, obj, '-k')
        save_fig(path, legend=False, **args)

    elif len(obj.columns.names) == 1:
        plot_ci = list(obj.columns) == ['beta', 'err']

        if plot_ci:
            plt.plot(obj.index, obj.beta, '-k')
            plt.plot(obj.index, obj.beta + obj.err, '--k')
            plt.plot(obj.index, obj.beta - obj.err, '--k')
        else:
            for c in obj.columns:
                s = obj[c].dropna()
                plt.plot(s.index, s, label=c)

        if diagonal:
            add_diagonal(obj)

        save_fig(path, legend=(not plot_ci), **args)

    else:
        dsets = obj.columns.get_level_values(0).unique()
        if dsets[0] == 'Humans':
            if dsets[1] in TRICOLOR:
                colors = TRICOLOR
            else:
                colors = COLORS
        else:
            colors = COLORS[1:]

        # plot together
        for i in range(len(dsets)):
            draw_simple(obj=obj, dsets=dsets, i=i, colors=colors)

        if diagonal:
            add_diagonal(obj)

        save_fig(path, **args)

        # plot separately
        kwargs = args.copy()  # type: dict
        kwargs['legend'] = False
        for i in range(len(dsets)):
            draw_simple(obj=obj, dsets=dsets, i=i, colors=COLORS)
            if name in [NORM, 'roc']:
                add_diagonal(obj)
            save_fig('{}_{}'.format(path, dsets[i]), **kwargs)

        # plot incrementally
        for i in range(len(dsets)):
            for j in range(i+1):
                draw_simple(obj=obj, dsets=dsets, i=j, colors=colors)

                if j == 0 and diagonal:
                    add_diagonal(obj)

            save_fig('{}_{}'.format(path, i), **args)
