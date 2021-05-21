import numpy as np
from matplotlib import pyplot as plt

from featnames import ACCEPT, REJECT, CON, NORM
from plots.const import BIN_TICKS, SLRBO_TICKS, COLORS
from plots.save import save_fig

from plots.util import add_diagonal, get_name


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
    elif name == 'counter':
        t = int(path[-1])
        args = dict(xlabel='Turn {}: Offer / list price'.format(t - 1),
                    ylabel='Turn {}: Pr(counter)'.format(t),
                    ylim=[0, 1])
    elif name == 'conacc':
        t = int(path[-1])
        dim = obj[0].index
        args = dict(xlabel='Turn {}: Concession'.format(t - 1),
                    ylabel='Turn {}: Pr(accept)'.format(t),
                    ylim=[0, 1], xlim=[min(dim), max(dim)],
                    legend_kwargs=dict(loc='upper left'))
    elif name.startswith('hist'):
        if name.endswith('acc1'):
            ylabel = 'Turn 1: Pr(accept)'
            ylim = [0, 1]
        elif name.endswith('con1'):
            ylabel = 'Turn 1: Offer / list price'
            ylim = [.5, .75]
        elif name.endswith('offers'):
            ylabel = 'Number of buyer offers'
            ylim = [1, 2]
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))
        labels = [0, 1, 3, 10, 30, 100, 300, 1000]
        ticks = list(obj[1].index) + list(np.log10(labels[1:]))
        args = dict(xticks=ticks, xticklabels=labels,
                    ylabel=ylabel, xlabel='Buyer experience',
                    ylim=ylim)
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
                save_fig('{}_{}'.format(path, dset.split(' ')[0]),
                         legend=False, **args)

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