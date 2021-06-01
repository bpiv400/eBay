import numpy as np
from matplotlib import pyplot as plt
from plots.const import BIN_TICKS_SHORT, FONTSIZE
from plots.util import save_fig, get_name


def draw_contour(s=None, vmin=0, vmax=1, inc=.01, zlabel=None, reverse=False):
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
    reverse, vmax, zlabel = False, 1, 'Turn 2: Pr(reject)'
    if name == 'interarrival':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival')
        reverse = True
        vmax = np.ceil(s.max())
        zlabel = 'Days between first two arrivals'
    elif name == 'rejdays':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival')
    elif name == 'accdays':
        args = dict(xlabel='Turn 1: Offer / list price',
                    ylabel='Days to first arrival')
        zlabel = 'Turn 2: Pr(accept)'
    elif name == 'rejbin':
        args = dict(yticks=np.log10(BIN_TICKS_SHORT),
                    yticklabels=BIN_TICKS_SHORT,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='List price ($)')
    elif name == 'slrrejbinacc':
        args = dict(yticks=np.log10(BIN_TICKS_SHORT),
                    yticklabels=BIN_TICKS_SHORT,
                    xlabel='Turn 1: Offer / list price',
                    ylabel='List price ($)')
        zlabel = 'Turn 3: Pr(accept)'

    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    # with colorbar
    draw_contour(s=s, vmax=vmax, reverse=reverse, zlabel=zlabel)
    save_fig('{}_bar'.format(path), legend=False, **args)

    # without colorbar
    draw_contour(s=s, vmax=vmax, reverse=reverse, zlabel=None)
    save_fig(path, legend=False, **args)
