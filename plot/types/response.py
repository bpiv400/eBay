import numpy as np
from matplotlib import pyplot as plt
from plot.util import save_fig, get_name


def response_plot(path, obj):
    name = get_name(path)
    if name == 'slrrejacc':
        args = dict(ylim=[0, .8],
                    xlabel='Turn 2: Offer / list price',
                    ylabel='Turn 3: Pr(accept)')
    elif name.startswith('hist'):
        if name.endswith('acc1'):
            ylabel = 'Turn 1: Pr(accept)'
            ylim = [0, 1]
        elif name.endswith('con1'):
            ylabel = 'Turn 1: Offer / list price'
            ylim = [.5, .75]
        elif name.endswith('offers'):
            ylabel = 'Buyer offers per thread'
            ylim = [1, 1.5]
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))
        labels = [0, 1, 3, 10, 30, 100, 300, 1000]
        ticks = list(obj[1].index) + list(np.log10(labels[1:]))
        args = dict(xticks=ticks, xticklabels=labels,
                    ylabel=ylabel, xlabel='Buyer experience',
                    ylim=ylim)
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    line, dots = obj  # object is tuple

    # plot line
    plt.plot(line.index, line.beta, '-', color='k')
    plt.plot(line.index, line.beta + 1.96 * line.err, '--', color='k')
    plt.plot(line.index, line.beta - 1.96 * line.err, '--', color='k')

    # plot dot(s)
    for i in range(len(dots.index)):
        x = dots.index[i]
        row = dots.iloc[i]
        plt.plot(x, row['beta'], 'o', color='k', ms=5)
        if 'err' in dots.columns:
            high = row['beta'] + 1.96 * row['err']
            low = row['beta'] - 1.96 * row['err']
            plt.plot([x, x], [low, high], '-', color='k')

    save_fig(path, legend=False, **args)
