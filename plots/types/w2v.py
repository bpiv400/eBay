import pandas as pd
from matplotlib import pyplot as plt
from plots.util import save_fig


def w2v_plot(path, df):
    plt.scatter(df.x, df.y, s=(df.s / 1e3), c=df.c, cmap=plt.get_cmap('plasma'))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=6)

    # add labels
    labels = pd.DataFrame()
    den = df.groupby('label')['s'].sum()
    for var in ['x', 'y']:
        labels[var] = (df[var] * df['s']).groupby(df.label).sum() / den
    for label in labels.index:
        plt.text(labels.loc[label, 'x'], labels.loc[label, 'y'], label,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=6)

    save_fig(path, xaxis=False, yaxis=False, legend=False)
