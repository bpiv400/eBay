import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe, itemfreq


def summary(data):
    hist = sns.distplot(data, kde=False)
    hist.savefig('data/hist.png')
    box = sns.boxplot(x=data)
    box.savefig('data/box.png')
    print(describe(data))


def main():
    data = pd.read_csv('data/train.csv')
    offr = data['offr_price'].value
    summary(offr)
    check_bins(offr)


def check_bins(offr):
    # creates num_obv x 2 np.array where the first column
    # corresponds to observations and the second column to their frequencies
    freq_table = itemfreq(offr)
    # reverse sort rows by descending order of the second column (ie frequency)
    freq_table = freq_table[freq_table[:, 1].argsort()[::-1]]

    # extract top percentiles of observations
    top_one = freq_table[0:int(freq_table.shape[0] * .01), 0]
    top_two = freq_table[0:int(freq_table.shape[0] * .02), 0]
    top_ten = freq_table[0:int(freq_table.shape[0] * .1), 0]
    top_five = freq_table[0:int(freq_table.shape[0] * .05), 0]

    # collect
    bin_array = [top_one, top_two, top_ten, top_five]
    bin_names = ['1%', '2%', '10%', '5%']
    # grab the high
    right = np.amax(offr)
    left = np.amin(offr)

    for bin_cents in bin_array:
        odd_bin_cents = bin_cents[::2]
        ev_bin_cents = bin_cents[1::2]
        last_odd = None
        if odd_bin_cents.size % 2 != 0:
            # extracting highest freq vals for even and odd freq vals
            last_odd = odd_bin_cents[(odd_bin_cents.size - 1)]
            last_even = ev_bin_cents[(ev_bin_cents.size - 1)]
            # finding highest midpoint
            highest_edge = (last_odd + last_even) / 2

            odd_bin_cents = odd_bin_cents[:(odd_bin_cents.size - 1)]
            low_edges = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)

            # remove lowest element from odd bin centers
            odd_bin_cents = odd_bin_cents[1:]
            # remove highest element from even bin centers
            ev_bin_cents = ev_bin_cents[:(ev_bin_cents.size - 1)]
            # find midpoint between every even freq val (except the highest)
            # and the odd freq val immediately above it
            high_edges = np.divide(np.add(ev_bin_cents, odd_bin_cents), 2)
            # adds highest edge to edge count
            high_edges = np.append(high_edges, highest_edge)
            edge_count = high_edges.size + low_edges.size + 2
            edges = np.zeros(edge_count)
            edges[0] = left
            edges[edge_count - 1] = right
            edges[1:(edge_count - 3):2] = low_edges
            edges[1:(edge_count - 2):2] = high_edges
        else:
            # find midpoint between every even  freq val and the odd freq val
            # immediately below
            low_edges = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
            # remove lowest element from odd bin centers
            odd_bin_cents = odd_bin_cents[1:]
            # remove highest element from even bin centers
            ev_bin_cents = ev_bin_cents[:(ev_bin_cents.size - 1)]

            # find midpoint between every even freq val (except the highest)
            # and the odd freq val immediately above it
            high_edges = np.divide(np.add(ev_bin_cents, odd_bin_cents), 2)

            # count total edges
            edge_count = low_edges + high_edges + 2

            # create edge vector
            edges = np.zeros(edge_count)
            edges[0] = left
            edges[edge_count - 1] = right
            edges[1:(edge_count - 2):2] = low_edges
            edges[2:(edge_count - 3):2] = high_edges
        hist = plt.hist(offr, )


if __name__ == '__main__':
    main()
