from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import pdf_plot, cdf_plot, survival_plot, grouped_bar
from processing.processing_consts import INTERVAL
from constants import PLOT_DIR, SIM, OBS, BYR_HIST_MODEL, MAX_DELAY, \
    HOUR, DAY, ARRIVAL_PREFIX
from featnames import CON, MSG, DELAY


def main():
    # load data
    p = load(PLOT_DIR + 'distributions.pkl')

    # offers per thread
    print('Offers per thread')
    df = pd.concat([p[SIM]['offers'].rename(SIM),
                    p[OBS]['offers'].rename(OBS)], axis=1)

    grouped_bar('p_offers', df,
                ylabel='Fraction of threads')

    # threads per offer
    print('Threads per offer')
    df = pd.concat([p[SIM]['threads'].rename(SIM),
                    p[OBS]['threads'].rename(OBS)], axis=1)

    grouped_bar('p_threads', df,
                ylabel='Fraction of listings')

    # arrival times pdf plot
    print('Arrivals')
    df = pd.concat([p[SIM][ARRIVAL_PREFIX].rename(SIM),
                    p[OBS][ARRIVAL_PREFIX].rename(OBS)], axis=1)
    df.index = df.index.values / (DAY / INTERVAL[1])

    xticks = np.arange(0, MAX_DELAY[1] / DAY, 3)

    pdf_plot('p_arrival', df,
             xticks=xticks,
             xlabel='Days since listing start',
             ylabel='Fraction of buyers, by hour')

    # byr hist model
    print('Buyer experience')
    df = pd.concat([p[SIM][BYR_HIST_MODEL].rename(SIM),
                    p[OBS][BYR_HIST_MODEL].rename(OBS)], axis=1)

    grouped_bar('p_{}'.format(BYR_HIST_MODEL), df,
                horizontal=True,
                xlabel='Fraction of buyers',
                ylabel='Count of past Best Offer threads')

    # delay cdf plot
    for turn in range(2, 8):
        print('Delay{}'.format(turn))
        df = pd.concat([p[SIM][turn][DELAY].rename(SIM),
                        p[OBS][turn][DELAY].rename(OBS)], axis=1)
        df.index = df.index.values / HOUR

        # survival plot
        df = 1 - df

        if turn in [2, 4, 6, 7]:
            threshold = MAX_DELAY[turn] / HOUR
            xticks = np.arange(0, threshold + 1e-8, 6)
            xlabel = 'Response time in hours'
        else:
            threshold = MAX_DELAY[turn] / DAY
            xticks = np.arange(0, threshold + 1e-8, 2)
            xlabel = 'Response time in days'

        survival_plot('p_{}{}'.format(DELAY, turn), df,
                      xlim=[0, threshold],
                      xticks=xticks,
                      xlabel=xlabel,
                      ylabel='Fraction with longer response time')

    # concession models
    for turn in range(1, 8):
        print('Con{}'.format(turn))
        df = pd.concat([p[SIM][turn][CON].rename(SIM),
                        p[OBS][turn][CON].rename(OBS)], axis=1)

        xticks = np.arange(0, 100 + 1e-8, 10)

        cdf_plot('p_{}{}'.format(CON, turn), df,
                 xlim=[0, 100],
                 xticks=xticks,
                 xlabel='Concession (%)',
                 ylabel='Cumulative share of offers')

    # msg models
    print('Messages')
    idx = range(1, 7)
    df = pd.DataFrame(index=idx)
    for k in [SIM, OBS]:
        df[k] = [p[k][i][MSG] for i in idx]

    grouped_bar('p_{}'.format(MSG), df,
                xlabel='Turn',
                ylabel='Message probability')


if __name__ == '__main__':
    main()
