import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse


def bins_from_common(offr):
    # creates num_obv x 2 np.array where the first column
    # corresponds to observations and the second column to their frequencies
    freq_table = np.unique(offr, return_counts=True)
    freq_table = np.column_stack([freq_table[0], freq_table[1]])
    # reverse sort rows by descending order of the second column (ie frequency)
    freq_table = freq_table[freq_table[:, 1].argsort()[::-1]]

    # extract top percentiles of observations
    top_one = freq_table[0:int(freq_table.shape[0] * .01), 0]
    top_two = freq_table[0:int(freq_table.shape[0] * .02), 0]
    top_ten = freq_table[0:int(freq_table.shape[0] * .1), 0]
    top_five = freq_table[0:int(freq_table.shape[0] * .05), 0]

    # collect
    # bin_array = [top_one, top_two, top_ten, top_five]
    # bin_names = ['1%', '2%', '10%', '5%']

    bin_array = [top_one, top_two]
    bin_names = ['1%', '2%']

    # grab the high
    right = np.amax(offr)
    left = np.amin(offr)

    for bin_cents, bin_name in zip(bin_array, bin_names):
        bin_cents = np.sort(bin_cents)
        odd_bin_cents = bin_cents[::2]
        ev_bin_cents = bin_cents[1::2]
        last_odd = None
        if bin_cents.size % 2 != 0:
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
            edges[1:(edge_count - 1):2] = low_edges
            edges[2:(edge_count):2] = high_edges
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
            edge_count = low_edges.size + high_edges.size + 2

            # create edge vector
            edges = np.zeros(edge_count)
            edges[0] = left
            edges[edge_count - 1] = right
            edges[1:(edge_count):2] = low_edges
            edges[2:(edge_count - 1):2] = high_edges
        plt.figure(figsize=(11.5, 6), dpi=300)
        n, _, _ = plt.hist(offr, bins=edges)
        plt.suptitle('Histogram of Rounded Data')
        plt.title('(Rounded to %s of values)' % bin_name)
        max_bin = np.argmax(n)
        med_bin = np.median(n)
        n = np.sort(n)[::-1]
        print(n[0:10])
        # zoomed in plot
        highest_edge = np.amin(edges[edges > 1000])
        edges = edges[edges <= highest_edge]
        zoom = offr[offr <= np.amax(edges)]
        plt.figure(figsize=(11.5, 6), dpi=300)
        n, _, _ = plt.hist(zoom, bins=edges)
        plt.suptitle('Zoomed Histogram of Rounded Data (<= $1000)')
        plt.title('(Rounded to %s of values)' % bin_name)
        plt.show()

        highest_edge = np.amin(edges[edges > 500])
        edges = edges[edges <= highest_edge]
        zoom = offr[offr <= np.amax(edges)]
        plt.figure(figsize=(11.5, 6), dpi=300)
        n, _, _ = plt.hist(zoom, bins=edges)
        plt.suptitle('Zoomed Histogram of Rounded Data (<= $500)')
        plt.title('(Rounded to %s of values)' % bin_name)
        plt.show()

        highest_edge = np.amin(edges[edges > 100])
        edges = edges[edges <= highest_edge]
        zoom = offr[offr <= np.amax(edges)]
        plt.figure(figsize=(11.5, 6), dpi=300)
        n, _, _ = plt.hist(zoom, bins=edges)
        plt.suptitle('Zoomed Histogram of Rounded Data (<= $100)')
        plt.title('(Rounded to %s of values)' % bin_name)
        plt.show()

        highest_edge = np.amin(edges[edges > 50])
        edges = edges[edges <= highest_edge]
        zoom = offr[offr <= np.amax(edges)]
        plt.figure(figsize=(11.5, 6), dpi=300)
        n, _, _ = plt.hist(zoom, bins=edges)
        plt.suptitle('Zoomed Histogram of Rounded Data (<= $50)')
        plt.title('(Rounded to %s of values)' % bin_name)
        plt.show()


# to be used with digitize(right = True)
# remove all vals from the associated array that are
# greater than high or less than low (exclusive both)
# before using

# currently doesn't produce a symmetric window around low and high points
def bins_from_midpoints(low, high, step):
    midpoints = np.array(list(range(low, high + step, step)))
    print(len(midpoints))
    odd_bin_cents = midpoints[::2]
    print(odd_bin_cents)
    ev_bin_cents = midpoints[1::2]
    if len(midpoints) % 2 == 0:
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        odd_bin_cents = odd_bin_cents[1:]
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high
        bins[:bin_count:2] = low_set
        bins[1:bin_count - 1:2] = high_set
    else:
        print('odd')
        odd_bin_cents = odd_bin_cents[:len(odd_bin_cents) - 1]
        print(odd_bin_cents)
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        ev_high = ev_bin_cents[len(ev_bin_cents) - 1]
        last_bin = (high + ev_high) / 2
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        odd_bin_cents = odd_bin_cents[1:]
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        high_set = np.append(high_set, last_bin)
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high
        bins[:bin_count - 1:2] = low_set
        bins[1:bin_count:2] = high_set
    return bins, midpoints

# bin all values in a particular column using an array of bin edges and bin midpoints


def digitize(df, bins, midpoints, colname):
    col_series = df[colname]
    vals = col_series.values
    val_bins = np.digitize(vals, bins, right=True)
    rounded_vals = midpoints[val_bins]
    df[colname] = rounded_vals
    return df


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--low', action='store', type=int)
    parser.add_argument('--high', action='store', type=int)
    parser.add_argument('--step', action='store', type=float)
    parser.add_argument('--dir', action='store', type=str)
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    low = args.low
    high = args.high
    step = args.step

    # initialize model
    model = sk.linear_model.LogisticRegression(penalty='L2', solver='lbfgs', max_iter=100,
                                               verbose=1, C=10)

    # load data slice
    read_loc = 'data/' + subdir + '/' + filename
    df = pd.read_csv(read_loc)

    # dropping columns that are not useful for prediction
    df.drop(columns=['Unnamed: 0', 'anon_item_id', 'anon_thread_id', 'anon_byr_id',
                     'anon_slr_id', 'src_cre_date', 'response_time', 'auct_start_dt',
                     'auct_end_dt', 'item_price', 'bo_ck_yn'], inplace=True)

    # fixing seller and buyer history
    # assumed that NAN slr and byr histories indicate having participated in
    # 0 best offer threads previously
    # deduced from the fact that no sellers / buyers have 0 as their history
    no_hist_slr = df[np.isnan(df['slr_hist'])].index
    df.loc[no_hist_slr, 'slr_hist'] = 0

    no_hist_byr = df[np.isnan(df['byr_hist'])].index
    df.loc[no_hist_byr, 'byr_hist'] = 0

    del no_hist_byr
    del no_hist_slr

    # setting percent feedback for 'green' sellers to median feedback
    # score
    never_sold = df[np.isnan(df['fdbk_pstv_src'].values)].index
    scores = df['fdbk_pstv_src'].values
    scores = scores[~np.isnan(scores)]
    med_score = np.median(scores)
    df.loc[never_sold, 'fdbk_pstv_src'] = med_score
    del scores
    del med_score
    del never_sold

    # and setting number of previous feedbacks received to 0
    never_sold = df[np.isnan(df['fdbk_score_src'].values)].index
    df.loc[never_sold, 'fdbk_score_src'] = 0
    del never_sold

    # setting initial feedback for green sellers to median
    # feedback score
    never_sold = df[np.isnan(df['fdbk_pstv_start'].values)].index
    scores = df['fdbk_pstv_start'].values
    scores = scores[~np.isnan(scores)]
    med_score = np.median(scores)
    df.loc[never_sold, 'fdbk_pstv_start'] = med_score
    del scores
    del med_score
    del never_sold

    # and setting number of previous feedbacks received to 0
    never_sold = df[np.isnan(df['fdbk_score_start'].values)].index
    df.loc[never_sold, 'fdbk_score_start'] = 0
    del never_sold

    # setting nans for re-listing indicator to 0 because there are
    # no zeros in the indicator column, implying that nans indicate 0
    not_listed = df[np.isnan(df['lstg_gen_type_id'].values)]
    df.loc[not_listed, 'lstg_gen_type_id'] = 0
    del not_listed

    # setting nans for mssg to 0 arbitrarily since only ~.001% of offers
    # have 0 for message
    no_msg = df[np.isnan(df['any_mssg'].values)].index
    df.loc[never_sold, 'any_mssg'] = 0
    del no_msg

    # dropping columns that have missing values for the timebeing
    df.drop(columns=['count2', 'count3', 'count4', 'ship_time_fastest', 'ship_time_slowest',
                     'ref_price2', 'ref_price3', 'ref_price4'], inplace=True)

    # dropping all threads that do not have ref_price1
    df.drop(df[np.isnan(df['ref_price1'].values)].index, inplace=True)

    # drop rows with offers below low threshold or start price above high threshhold
    low_threads = df[df[df['offr_price'] < low].index,
                     'unique_thread_id'].values
    high_threads = df[df[df['start_price_usd'] > high].index,
                      'unique_thread_id'].values
    outlier_threads = np.unique(np.array(low_threads, high_threads))
    thread_ids = df[df['unique_thread_id'].isin(outlier_threads)].index
    df.drop(thread_ids, inplace=True)
    bins, midpoints = bins_from_midpoints(low, high, step)
    df = digitize(df, bins, midpoints, 'offr_price')
    df = digitize(df, bins, midpoints, 'resp_offr')
    save_loc = 'data/' + 'curr_exp' + \
        '/' + filename.replace('_feats2.csv', '.csv')
    df.to_csv(save_loc, index_label=False)


if __name__ == '__main__':
    bins, midpoints = bins_from_midpoints(1, 10, 1)
    print('%d, %d' % (len(bins), len(midpoints)))
