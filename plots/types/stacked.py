from matplotlib import pyplot as plt

from featnames import CON
from plots.const import COLORS
from plots.save import save_fig
from plots.util import get_name


def stacked_plot(path, df):
    name = get_name(path)
    if name == CON:
        args = dict(xlabel='Concession / list price', ylabel='',
                    xlim=[.45, 1],
                    yticks=range(len(df.columns)), yticklabels=df.columns,
                    legend_outside=True, legend_kwargs=dict(title='Turn'))
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    df.transpose().plot.barh(stacked=True, color=COLORS[1:3])
    plt.gca().invert_yaxis()
    save_fig(path, **args)
