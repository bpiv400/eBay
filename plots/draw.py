import argparse
import os
from plots.util import cdf_plot, grouped_bar, action_plot
from utils import unpickle
from constants import PLOT_DIR, FIG_DIR


def draw_counts(s, path=None):
    if 'offers' in path:
        den = 'threads'
        ylim = [0., .6]
    else:
        den = 'listings'
        ylim = [0, 1]
    grouped_bar(path, s,
                ylim=ylim,
                ylabel='Fraction of {}'.format(den))


def draw_cdf(df, path=None):
    name = path.split('/')[-1]
    if 'price' in name:  # for log scale
        df = df.loc[df.index > 0, :]
        xlim = [1, 1000]
        xlabel = 'Sale price'
        logx = True
    elif 'norm' in name:
        xlim = [.01, 1]
        xlabel = 'Sale price / list price'
        logx = False
    elif 'months' in name:
        xlim = [0, 1]
        xlabel = 'Fraction of month'
        logx = False
    else:
        raise NotImplementedError()

    cdf_plot(path, df,
             xlim=xlim,
             xlabel=xlabel,
             ylabel='Cumulative share of listings',
             logx=logx,
             legend_kwargs=dict(loc='upper left'))


def main():
    # subset
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--subset', type=str)
    args = parser.parse_args()
    prefix, subset = args.prefix, args.subset

    path = PLOT_DIR + prefix
    folder = '{}/'.format(prefix)
    if subset is not None:
        path += '_{}'.format(subset)
        folder += '{}/'.format(subset)

    if not os.path.isdir(FIG_DIR + folder):
        os.makedirs(FIG_DIR + folder)

    d = unpickle('{}.pkl'.format(path))
    for k, v in d.items():
        print(k)
        if k == 'action':
            for name in d[k].keys():  # wave plots
                for t, df in d[k][name].items():
                    path = '{}/action_{}_{}'.format(folder, name, t)
                    action_plot(path, df, turn=t)
        else:
            path = '{}/{}'.format(folder, k)
            if k.startswith('num'):  # bar charts
                draw_counts(v, path=path)
            elif k.startswith('cdf'):  # cdf plots
                draw_cdf(v, path)


if __name__ == '__main__':
    main()
