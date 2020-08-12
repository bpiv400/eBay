import argparse
from plots.util import cdf_plot, grouped_bar, action_plot, con_plot
from utils import unpickle
from constants import PLOT_DIR
from featnames import SLR


def draw_counts(s, path=None):
    if 'offers' in path:
        den = 'threads'
    else:
        den = 'trajectories'
    grouped_bar(path, s,
                ylim=[0, 1],
                ylabel='Fraction of {}'.format(den))


def draw_cdf(df, path=None):
    if 'price' in path:  # for log scale
        df = df.loc[df.index > 0, :]
        xlim = [1, 1000]
        xlabel = 'Sale price'
        logx = True
    else:
        xlim = [0.2, 1]
        xlabel = 'Sale price / list price'
        logx = False

    cdf_plot(path, df,
             xlim=xlim,
             xlabel=xlabel,
             ylabel='Cumulative share of listings',
             logx=logx,
             legend_kwargs=dict(loc='upper left'))


def draw_action(df, path=None):
    action_plot(path, df, byr=False)


def draw_con(df, path):
    con_plot(path, df)


def main():
    # flag for relisting environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--relist', action='store_true')
    relist = parser.parse_args().relist
    suffix = 'relist' if relist else 'norelist'
    folder = '{}_{}/'.format(SLR, suffix)

    d = unpickle(PLOT_DIR + '{}_{}.pkl'.format(SLR, suffix))
    for k, v in d.items():
        print(k)
        if k == 'action':
            for name in d[k].keys():  # wave plots
                for t, df in d[k][name].items():
                    path = folder + 'action_{}_{}'.format(name, t)
                    draw_action(df, path=path)
        elif k == 'con':
            for t, df in d[k].items():  # average concession
                path = folder + 'con_{}'.format(t)
                draw_con(df, path=path)
        else:
            path = folder + '{}'.format(k)
            if k.startswith('num'):  # bar charts
                draw_counts(v, path=path)
            elif k.startswith('cdf'):  # cdf plots
                draw_cdf(v, path)


if __name__ == '__main__':
    main()
