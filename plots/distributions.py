from compress_pickle import load
import numpy as np
from plots.plots_utils import grouped_bar, overlapping_bar, save_fig
from processing.processing_consts import NUM_OUT
from constants import PLOT_DIR, CON_MULTIPLIER, CON_MODELS, MSG_MODELS


def main():
    # load data
    num_offers = load(PLOT_DIR + 'num_offers.pkl')
    num_threads = load(PLOT_DIR + 'num_threads.pkl')
    p = load(PLOT_DIR + 'distributions.pkl')

    # offers per thread
    labels = [str(i) for i in num_offers.index]
    y = num_offers.to_dict(orient='list')

    grouped_bar(labels, y)
    save_fig('num_offers',
             legend='upper right',
             xlabel='Offers',
             ylabel='Fraction of threads')

    # threads per offer
    keep = 4
    num_threads.iloc[keep, :] = num_threads.iloc[keep:].sum(axis=0)
    num_threads = num_threads[:keep+1]

    labels = [str(i) for i in num_threads.index]
    labels[-1] += '+'
    y = num_threads.to_dict(orient='list')

    grouped_bar(labels, y)
    save_fig('num_threads',
             legend='upper right',
             xlabel='Threads',
             ylabel='Fraction of listings',
             gridlines=False)

    # models with binary outcomes
    models = MSG_MODELS + [CON_MODELS[-1]]
    y = dict()
    for k in ['simulated', 'observed']:
        y[k] = [p[m][k][-1] for m in models]

    labels = ['$\\texttt{{msg}}_{}$'.format(i) for i in range(1, 7)]
    labels += ['$\\texttt{{con}}_7$']

    grouped_bar(labels, y)
    save_fig('p_binary',
             legend='upper left',
             ylabel='Probability of message / acceptance',
             gridlines=False)

    # concession models
    for m in CON_MODELS[:-1]:
        ticks = np.arange(0, NUM_OUT[m], 10)
        labels = ticks / CON_MULTIPLIER
        overlapping_bar(p[m], ticks=ticks, labels=labels)
        save_fig('p_{}'.format(m),
                 xlabel='Concession',
                 gridlines=False)


if __name__ == '__main__':
    main()
