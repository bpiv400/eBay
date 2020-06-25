from compress_pickle import load
import numpy as np
import pandas as pd
from plots.util import continuous_pdf, cdf_plot, survival_plot, \
    grouped_bar
from inputs.const import ARRIVAL_INTERVAL
from constants import PLOT_DIR, SIM, OBS, BYR_HIST_MODEL, MAX_DELAY, \
    HOUR, DAY, ARRIVAL
from featnames import CON, MSG, DELAY


def construct_df(p, key, turn=None):
    if turn is None:
        return pd.concat([p[SIM][key].rename(SIM),
                          p[OBS][key].rename(OBS)], axis=1)
    else:
        return pd.concat([p[SIM][turn][key].rename(SIM),
                          p[OBS][turn][key].rename(OBS)], axis=1)


def draw_thread_offer(p, suffix=''):
    for key in ['offers', 'threads']:
        name = '{}{}'.format(key, suffix)
        print(name)
        df = construct_df(p, key)

        den = 'threads' if key == 'offers' else 'listings'
        grouped_bar(name, df,
                    ylim=[0, 0.6],
                    ylabel='Fraction of {}'.format(den))


def draw_arrival(p, suffix=''):
    # arrival model
    name = '{}{}'.format(ARRIVAL, suffix)
    print(name)
    df = construct_df(p, ARRIVAL)
    df.index = df.index.values / (DAY / ARRIVAL_INTERVAL)

    xticks = np.arange(0, MAX_DELAY[1] / DAY, 3)

    continuous_pdf(name, df,
                   xticks=xticks,
                   xlabel='Days since listing start',
                   ylabel='Fraction of buyers, by hour')

    # byr hist model
    name = '{}{}'.format(BYR_HIST_MODEL, suffix)
    print(name)
    df = construct_df(p, BYR_HIST_MODEL)

    grouped_bar(name, df,
                horizontal=True,
                xlabel='Fraction of buyers',
                ylabel='Count of past Best Offer threads')


def draw_delay(p, suffix='', turns=range(2, 8)):
    # delay cdf plot
    for turn in turns:
        name = '{}{}{}'.format(DELAY, turn, suffix)
        print(name)
        df = construct_df(p, DELAY, turn=turn)

        # survival plot
        df = 1 - df

        if turn in [2, 4, 6, 7]:
            df.index = df.index.values / HOUR
            threshold = MAX_DELAY[turn] / HOUR
            xticks = np.arange(0, threshold + 1e-8, 6)
            xlabel = 'Response time in hours'
        else:
            df.index = df.index.values / DAY
            threshold = MAX_DELAY[turn] / DAY
            xticks = np.arange(0, threshold + 1e-8, 2)
            xlabel = 'Response time in days'

        survival_plot(name, df,
                      xlim=[0, threshold],
                      xticks=xticks,
                      xlabel=xlabel,
                      ylabel='Fraction with longer response time')


def draw_con(p, suffix='', turns=range(1, 8)):
    for turn in turns:
        name = '{}{}{}'.format(CON, turn, suffix)
        print(name)
        df = construct_df(p, CON, turn=turn)

        xticks = np.arange(0, 100 + 1e-8, 10)

        cdf_plot(name, df,
                 xlim=[0, 100],
                 xticks=xticks,
                 xlabel='Concession (%)',
                 ylabel='Cumulative share of offers')


def draw_msg(p, suffix=''):
    name = '{}{}'.format(MSG, suffix)
    print(name)
    idx = range(1, 7)
    df = pd.DataFrame(index=idx)
    for k in [SIM, OBS]:
        df[k] = [p[k][i][MSG] for i in idx]

    grouped_bar(name, df,
                xlabel='Turn',
                ylabel='Share of eligible offers')


def main():
    # load data
    p = load(PLOT_DIR + 'p.pkl')

    # draw plots
    draw_thread_offer(p)
    draw_arrival(p)
    draw_delay(p)
    draw_con(p)
    draw_msg(p)


if __name__ == '__main__':
    main()
