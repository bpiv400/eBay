from compress_pickle import load
import numpy as np
from plots.plots_utils import grouped_bar, save_fig
from constants import PLOT_DIR


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
    d = load(PLOT_DIR + 'distributions.pkl')



# number of threads per listing
# num_threads = load(PLOT_DIR + 'distributions.pkl')


if __name__ == '__main__':
    main()
