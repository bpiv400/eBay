import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from featnames import BYR, SLR, MSG, NORM, REJECT, EXP
from plots.const import COLORS
from plots.save import save_fig
from plots.util import get_name


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
        lower, upper = np.floor(obj.min()) - 1, np.ceil(obj.max())
        ylim = [lower, upper]
        ylabel = '{} ($)'.format('Discount' if name.endswith('2')
                                 else 'Payoff')
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
    elif name == 'lambda':
        args = dict(xlabel='List price multiplier $(\\lambda)$')
        last = path.split('/')[-1].split('_')[2]
        if last == 'offers':
            args['ylabel'] = 'Number of buyer offers'
            args['ylim'] = [1, 3]
        elif last == 'first':
            args['ylabel'] = 'Turn 1: Offer / list price'
            args['ylim'] = [.49, .66]
        elif last == 'bin':
            args['ylabel'] = 'Turn 1: Pr(accept)'
            args['ylim'] = [0, .3]
        elif last == 'counter':
            t = path.split('_')[-1]
            args['ylabel'] = 'Turn {}: Pr(counter)'.format(t)
            args['ylim'] = [0, 1]
        else:
            raise NotImplementedError('Invalid name: {}'.format(last))
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is pd.Series:
        if name == 'lambda':
            plt.gca().axhline(y=obj.loc['Humans'], label='Humans',
                              c='k', ls='--', lw=1)
            obj = obj.drop('Humans')
            obj.plot.bar(rot=45, color='gray', label='')
        else:
            obj.plot.bar(rot=0, color=COLORS[0])
            plt.gca().axhline(c='k', lw=1)
    else:
        rot = 45 if len(obj.columns) > 3 else 0
        obj.plot.bar(rot=rot, color=COLORS)
    save_fig(path, xticklabels=obj.index, gridlines=False, **args)