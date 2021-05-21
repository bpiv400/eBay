import numpy as np
from matplotlib import pyplot as plt
from plots.const import BIN_TICKS_SHORT, FONTSIZE
from plots.save import save_fig
from plots.util import get_name


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
                    zlabel='Turn 2: Pr(reject)')
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
        args = dict(yticks=np.log10(BIN_TICKS_SHORT),
                    yticklabels=BIN_TICKS_SHORT,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='List price ($)',
                    zlabel=zlabel if suffix in ['data', name] else None)
    elif name in ['offer2binwalk', 'offer2binacc']:
        turn = int(name[5])
        zlabel = 'walk' if name.endswith('walk') else 'accept'
        args = dict(yticks=np.log10(BIN_TICKS_SHORT),
                    yticklabels=BIN_TICKS_SHORT,
                    xlabel='Turn {}: Offer / list price'.format(turn),
                    ylabel='List price ($)',
                    zlabel='Turn {}: Pr({})'.format(turn+1, zlabel),
                    vmax=.5)
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # with colorbar
    draw_contour(s=s, **args)
    save_fig('{}_bar'.format(path), legend=False, **args)

    # without colorbar
    args['zlabel'] = None
    draw_contour(s=s, **args)
    save_fig(path, legend=False, **args)
