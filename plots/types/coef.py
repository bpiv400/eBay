from matplotlib import pyplot as plt

from plots.save import save_fig
from plots.util import get_name


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