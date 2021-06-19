import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from constants import MAX_DELAY_TURN, HOUR, EPS
from featnames import DELAY, CON, NORM
from plots.const import COLORS
from plots.util import save_fig, get_name


def add_line(x=None, y=None):
    if type(x) is list:
        plt.plot(x, [y, y], '-k', lw=1)
    elif type(y) is list:
        plt.plot([x, x], y, '-k', lw=1)
    else:
        raise ValueError('x or y must a list.')


def cdf_plot(path, obj):
    name = get_name(path)
    den = 'listings'

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
                    xlabel='Sale price / list price',
                    legend_outside=('heuristic' in path))
        if name == 'salenorm':
            den = 'sales'
    elif name in ['days', 'arrivaltime']:
        args = dict(xlim=[0, 1],
                    xlabel='Fraction of listing window')
        if name == 'arrivaltime':
            den = 'arrivals'
    elif name == 'hist':
        args = dict(xlim=[0, 250],
                    xlabel='Prior Best Offer threads for buyer')
        den = 'threads'
    elif name == DELAY:
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
    elif name == 'realval':
        args = dict(xlim=[0, 1],
                    xlabel='Realized value')
    elif name == 'bin':
        ticks = [10] + list(range(200, 1001, 200))
        args = dict(xlim=[9.95, 1000], xlabel='List price',
                    xticks=ticks,
                    xticklabels=['${}'.format(t) for t in ticks])

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
                    save_fig('{}_{}'.format(path, label.split(' ')[0]),
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
