from plots.util import cdf_plot, grouped_bar, response_plot
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
    if 'sale_price' in path:  # for log scale
        df = df.loc[df.index > 0, :]
        xlim = [1, 1000]
        xlabel = 'Sale price'
        logx = True
    else:
        xlim = [0.4, 1]
        xlabel = 'Sale price / list price'
        logx = False

    cdf_plot(path, df,
             xlim=xlim,
             xlabel=xlabel,
             ylabel='Cumulative share of listings',
             logx=logx,
             legend_kwargs=dict(loc='upper left'))


def draw_response(df, path=None):
    response_plot(path, df, byr=False)


def main():
    # seller agent
    d = unpickle(PLOT_DIR + '{}.pkl'.format(SLR))
    for k, v in d.items():
        if k != 'y_hat':
            path = '{}/{}'.format(SLR, k)
            f = draw_cdf if 'sale' in k else draw_counts
            f(v, path=path)
        else:
            for delta in d[k].keys():
                for t, df in d[k][delta].items():
                    path = '{}/response_{}_{}'.format(SLR, delta, t)
                    draw_response(df, path=path)


if __name__ == '__main__':
    main()
