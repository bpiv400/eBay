from compress_pickle import load
import numpy as np
from plots.plots_utils import grouped_bar, save_fig
from processing.processing_consts import NUM_OUT
from constants import PLOT_DIR, MODELS


def main():
    # offers per thread
    num_offers = load(PLOT_DIR + 'num_offers.pkl')

    labels = [str(i) for i in num_offers.index]
    y = num_offers.to_dict(orient='list')

    grouped_bar(labels, y)
    save_fig('num_offers',
             legend='upper right',
             xlabel='Offers',
             ylabel='Fraction of threads')

    # threads per offer
    num_threads = load(PLOT_DIR + 'num_threads.pkl')

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
             ylabel='Fraction of listings')

    # models with binary outcomes
    p = load(PLOT_DIR + 'distributions.pkl')

    labels = [m for m in MODELS if NUM_OUT[m] == 1]
    y = dict()
    for k in ['simulated', 'observed']:
        y[k] = [y[m][k][-1] for m in labels]

    grouped_bar(labels, y)
    save_fig('')


if __name__ == '__main__':
    main()
