import pandas as pd
from matplotlib import pyplot as plt
from plots.util import save_fig, get_name
from plots.const import COLORS
from featnames import BYR, MSG, REJECT


def bar_plot(path, obj):
    name = get_name(path)
    if name == 'offers':
        args = dict(ylim=[0., .6],
                    xlabel='Turn of last offer',
                    ylabel='Fraction of threads')
    elif name == 'threads':
        xlabel = 'Buyer number' if path.startswith(BYR) else 'Number of threads'
        args = dict(ylim=[0, 1],
                    xlabel=xlabel,
                    ylabel='Fraction of listings')
    elif name == MSG:
        args = dict(ylim=[0, .5],
                    xlabel='Turn',
                    ylabel='Fraction of eligible offers')
    elif name == 'training':
        baserate = obj['Baserate']
        obj.drop('Baserate', inplace=True)
        args = dict(ylim=[baserate, None], legend=False,
                    xlabel='', ylabel='')
    elif name == REJECT:
        args = dict(xlabel='Turn', ylabel='Pr(reject)')
    elif name == 'lambda':
        args = dict(xlabel='List price multiplier $(\\lambda)$')
        last = path.split('/')[-1].split('_')[2]
        if last == 'offers':
            args['ylabel'] = 'Number of buyer offers'
            args['ylim'] = [1, 3]
        elif last == 'bin':
            args['ylabel'] = 'Turn 1: Pr(accept)'
            args['ylim'] = [0, .3]
        else:
            raise NotImplementedError('Invalid name: {}'.format(last))
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    if type(obj) is pd.Series:
        if name == 'lambda':
            plt.gca().axhline(y=obj.loc['Humans'], label='Humans',
                              c='k', ls='--', lw=1)
            obj = obj.drop('Humans')
            obj.plot.bar(rot=0, color='gray', label='')
        else:
            obj.plot.bar(rot=0, color=COLORS[0])
            plt.gca().axhline(c='k', lw=1)
    else:
        rot = 45 if len(obj.columns) > 3 else 0
        obj.plot.bar(rot=rot, color=COLORS)
    save_fig(path, xticklabels=obj.index, gridlines=False, **args)
