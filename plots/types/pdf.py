import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from constants import EPS
from plots.save import save_fig
from plots.util import get_name


def pdf_plot(path, obj):
    name = get_name(path)
    if name == 'arrival':
        args = dict(xlim=[0, 1],
                    xlabel='Fraction of listing window')
    elif name == 'interarrival':
        args = dict(xlim=[0, 48], xlabel='Hours',
                    xticks=np.arange(0, 48 + EPS, 12))
    elif name == 'values':
        args = dict(xlim=[0, 1],
                    xlabel='Value / list price')
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    legend = type(obj) is pd.DataFrame
    if not legend:
        plt.plot(obj.index, obj, color='k')
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max()],
                 gridlines=False,
                 **args)

    elif 'Data' in obj.columns:
        s = obj['Data']

        # plot separately
        plt.plot(s.index, s, color='k')
        save_fig('{}_Data'.format(path),
                 legend=False,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)

        # plot together
        plt.plot(s.index, s, label='Data', color='k')
        df = obj.drop('Data', axis=1)
        for i in range(len(df.columns)):
            c = df.columns[i]
            plt.plot(df.index, df[c], label=c)
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)

    else:
        for i in range(len(obj.columns)):
            c = obj.columns[i]
            plt.plot(obj.index, obj[c], label=c)
        save_fig(path,
                 legend=legend,
                 yaxis=False,
                 ylim=[0, obj.max().max()],
                 gridlines=False,
                 **args)