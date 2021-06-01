import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from featnames import ACCEPT, REJECT, NORM
from plots.const import BIN_TICKS, TRICOLOR, COLORS, SLRBO_TICKS
from plots.save import save_fig
from plots.util import get_name, add_diagonal


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
        diagonal = True
    elif name in [ACCEPT, REJECT]:
        args = dict(xlim=[.4, .9], ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Pr({})'.format(name))
    elif name == NORM:
        args = dict(xlim=[.4, 1], ylim=[.4, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Offer / list price')
    elif name == 'slrrejaccbin':
        args = dict(ylim=[0, 1],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Turn 3: Pr(accept)')
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
        diagonal = True
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
    elif name == 'discountbin':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[.1, .35],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount')
    elif name == 'avgdiscountbin':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[0, 50],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount / buyer offer ($)')
    elif name == 'discount':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[0, .4],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Discount / list price')
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
    elif name == 'listfirst':
        args = dict(xlim=[np.log10(20), max(obj.index)], ylim=[.5, .75],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Turn 1: Offer  / list price')
    elif name == 'listrej':
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[0, 1],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Pr(reject)')
    elif name == 'listcon':
        args = dict(xlim=[1, 2.5], ylim=[0, 2],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Average concession')
    elif name.startswith('listcounter'):
        t = path.split('_')[-1]
        args = dict(xlim=[min(obj.index), max(obj.index)], ylim=[0, 1],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Turn {}: Pr(counter)'.format(t))
    elif name == 'conpath':
        args = dict(xlabel='Turn', ylabel='Offer / list price',
                    xlim=[1, 6], ylim=[.5, 1])
    elif name == 'conrepeat':
        args = dict(xlim=[.4, .9], ylim=[0, .25],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='',
                    legend_kwargs=dict(title='Turn 2'))
    elif name == 'listsale':
        args = dict(xlim=[1, max(obj.index)], ylim=[.5, 1],
                    xticks=np.log10(BIN_TICKS),
                    xticklabels=BIN_TICKS,
                    xlabel='List price ($)',
                    ylabel='Negotiated sale price / list price')
    elif name == 'offer2norm':
        bounds = [min(obj.index), max(obj.index)]
        args = dict(xlim=bounds, ylim=bounds,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Turn 2: Offer / list price')
        diagonal = True
    elif name == 'offer2time':
        args = dict(xlim=[min(obj.index), max(obj.index)],
                    ylim=[0, 1],
                    xlabel='Turn 1: Offer / list price',
                    ylabel='Pr(Seller responds within 6 hours)')
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
