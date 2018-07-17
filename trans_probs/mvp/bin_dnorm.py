# load packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import pickle
import argparse
import math

# to be used with digitize(right = True)
# remove all vals from the associated array that are
# greater than high or less than low (exclusive both)
# before using


def bin_times_from_midpoints(step):
    '''
    Description: Creates uniformly spaced bin midpoints 
    for times based on a step given in minutes for 
    times from 0 to 48 hours and outputs these as well
    as an array of right sides for these bins to 
    be used in np.digitize
    Input:
        step: integer giving number of minutes between
        midpoints
    Output: length 2 tuple containing: bins, midpoints
    arrays
    '''
    # create evenly spaced array for midpoints
    midpoints = np.arange(0, (48*60 + step), step)
    # round to the nearest minute
    midpoints = np.around(midpoints, 0)
    # grab high from midpoints
    max_midpoint = np.amax(midpoints)
    half_bin_width = step / 2
    bins = np.arange(half_bin_width, max_midpoint +
                     half_bin_width * 2, half_bin_width * 2)
    if bins.size != midpoints.size:
        raise ValueError('The number of midpoints and bins should be the same')
    return bins, midpoints


def bins_from_midpoints(low, high, step):
    '''
    Description: Creates uniformly spaced bin midpoints and outputs these
    as well as an array of right sides for these bins to be used in np.digitize(...)
    later for rounding
    Input:
        low: float, lowest tolerated midpoint
        high: float, highest tolerated midpoint
        step: difference between successive midpoints
    Output:
        Tuple of 2 np.array's where the first gives the right side of each
        bin created and the second gives the corresponding midpoints for each
        of these bins
    '''
    # create array of midpoints
    midpoints = np.arange(low, high + step, step)
    # grab every other midpoint starting with 0
    odd_bin_cents = midpoints[::2]
    # grab every other midpoint starting with 1
    ev_bin_cents = midpoints[1::2]
    # if there are an even number of midpoints
    if len(midpoints) % 2 == 0:
        # find the midpoint between each odd and even midpoint\
        # (taking midpoint of each odd center and the even midpoint
        # immediately above it)
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        # remove the low
        odd_bin_cents = odd_bin_cents[1:]
        # remove the high
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        # find midpoints between each remaining elementwise pair,
        # corresponds to taking the midpoint between each element in
        # odd midpoint array and the element immediately below it
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        # create bin array
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high
        bins[:bin_count:2] = low_set
        bins[1:bin_count - 1:2] = high_set
    else:
        # remove the greatest bin center, corresopnding to high
        odd_bin_cents = odd_bin_cents[:len(odd_bin_cents) - 1]
        # find the midpoint between each odd and even midpoint\
        # (taking midpoint of each odd center and the even midpoint
        # immediately above it)
        low_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        # remove the last midpoint from the even array, ie the second
        # highest midpoint
        ev_high = ev_bin_cents[len(ev_bin_cents) - 1]
        # find the midpoint between the highest midpoint and the second highest
        last_bin = (high + ev_high) / 2
        ev_bin_cents = ev_bin_cents[:len(ev_bin_cents) - 1]
        # remove the lowest bin from the odd array
        odd_bin_cents = odd_bin_cents[1:]
        # find midpoints between each remaining elementwise pair,
        # corresponds to taking the midpoint between each element in
        # odd midpoint array and the element immediately below it
        high_set = np.divide(np.add(odd_bin_cents, ev_bin_cents), 2)
        # append the greatest internal bin found onto high_set
        high_set = np.append(high_set, last_bin)
        # construct the resulting vector
        bin_count = len(low_set) + len(high_set) + 1
        bins = np.zeros(bin_count)
        bins[bin_count - 1] = high + step
        bins[:bin_count - 1:2] = low_set
        bins[1:bin_count:2] = high_set
    return bins, midpoints

# to be used with digitize(right = True)
# remove all vals from the associated array that are
# greater than high or less than low (exclusive both)
# before using


def bins_from_common(offr, percent=None, num=None):
    '''
    Description: Given a 1-dimensional np.aray of offers, create
    a vector of the x most common values or the top y% of values
    and an array giving the right hand side of each resulting bin
    as a tuple (bins, midpoints)
    Input:
        offr: np.array of values
        percent: float < 1 denoting the percentage of most common
        values to be extracted and binned
        num: int > 1 denoting the number of most common values to be used
    Output:
        tuple of np.array's of equal size where the first denotes the
        right edge of each bin and the second denotes the midpoint
        of each bin (ie the value to which offers will later be rounded)
    '''

    # creates num_obv x 2 np.array where the first column
    # corresponds to observations and the second column to their frequencies
    freq_table = np.unique(offr, return_counts=True)
    freq_table = np.column_stack([freq_table[0], freq_table[1]])
    # reverse sort rows by descending order of the second column (ie frequency)
    freq_table = freq_table[freq_table[:, 1].argsort()[::-1]]

    # extract top percentiles of observations
    if percent is not None:
        bin_cents = freq_table[0:int(freq_table.shape[0] * percent), 0]
    else:
        bin_cents = freq_table[0:num, 0]

    # grab the high
    right = np.amax(offr)
    # grab the low
    left = np.amin(offr)

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
    # remove low, since output is piped to np.digitize(..right = True)
    edges = edges[1:]
    # currently, high bin equals the highest rounding point
    # this is fine but not ideal since we would prefer a symmetric bin around each
    # binning value
    # therefore, we increase the highest bin by (right side high bin - right side next highest bin),
    # thereby creating a symmetric range
    # we leave the highest midpoint value unchanged, because we would still like to round
    # to this value
    left_width = edges[len(edges) - 1] - edges[len(edges) - 2]
    edges[len(edges) - 1] = edges[len(edges) - 1] + left_width

    # the lowest bin's right edge is the first element in the edges array & digitize (as we call it)
    # only takes in right edges for each bin
    # the digitize function places all values lower than this right edge into the first
    # bin...as a result, we don't need to similarly extend the left edge here
    # however, we want to keep all threads in the data set with offers in range [left_edge_low_bin, right_edge_high_bin]
    # as a result, we need to calculate this left edge and use it to subset data, so we don't just
    # chop it at the bin center
    # this is done after getting function output in main thread
    return edges, bin_cents


def digitize(df, bins, midpoints, colname):
    '''
    Description: Round values in a DataFrame column into
    n particular bins
    Inputs:
        df: dataframe containing column whose values will
        be rounded
        bins: np.array of size n denoting the right hand side
        of each bin into which values will be rounded
        midpoints: np.array of size n denoting the values to which
        each the values placed into bin i will be rounded
        colname: name of the column in df whose being operated on
    Output: df with rounded values in df[colname]
    '''
    col_series = df[colname]
    midpoints = np.array(midpoints)
    vals = col_series.values
    ind = col_series.index
    val_bins = np.digitize(vals, bins, right=True)
    rounded_vals = midpoints[val_bins]
    df[colname] = pd.Series(rounded_vals, index=ind)
    return df


def get_resp_offr(turn):
    '''
    Description: Determines the name of the response column given the name of the last observed turn
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    if turn != 'start_price_usd':
        turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = 'b0'
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
    return resp_col


def get_resp_time(turn):
    '''
    Description: Determines the name of the response column given the 
    name of the last observed turn
    for time models
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = None
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'time_%s' % resp_turn
    return resp_col


def get_diff(midpoints, abs_tol=None, perc_tol=None):
    '''
    Description: Calculate the difference between the midpoint of
    the ith bin and i+1 bin for all bins 0 to len(midpoints) - 2,
    and give whether each is less than the hyperparameter for
    tolerance
    Inputs:
        midpoints: array where element i gives the midpoint (rounding point)
        of the ith bin
        abs_tol: minimum difference tolerance in dollars
        perc_tol: minimum difference tolerance given as a percent of higher bin (high-low)/high
    Output: boolean 1-dimensional ndarray where element i gives whether
    diff(i+1, i) was less than the threshold
    '''
    if perc_tol is not None:
        # divide each midpoint by the next midpoint giving ratios of lower/higher
        # for each midpoint except the last, for which no such value can be calculated
        diff = [midpoints[i] / midpoints[i+1]
                for i in range(len(midpoints) - 1)]
        diff = np.array(diff)
        # find 1 - low/high, which corresponds to (high-low)/high
        # value will be lower when difference between low and high is lower
        diff = 1 - diff
        # set diff to whether each was less than the required threshold
        diff = diff < perc_tol

    else:
        # simply take the difference between each successive midpoint if absolute
        # tolerance is used
        diff = [midpoints[i+1] - midpoints[i]
                for i in range(len(midpoints) - 1)]
        diff = np.array(diff)
        diff = diff < abs_tol
    return diff


def squash(bins, midpoints, low, high, abs_tol=None, perc_tol=None):
    '''
    Description: Forces midpoints of all bins to be between [low, high] and
    forces midpoints to be at least some tolerance distance apart
    If abs_tol is not none, this value represents an absolute tolerance in dollars
    If perc_tol is not none, this value represents a percent tolerance--(1-lower/higher)

    Input:
        bins: an array of floats where each value gives the right side limit of the
        ith bin
        midpoints: an array of floats where each value gives the midpoint (the value to which
        offers will be rounded)
        low: float giving lowest tolerated midpoint
        high: float giving highest tolerated midpoint
        abs_tol: minimum difference tolerance in dollars
        perc_tol: minimum difference tolerance given as a percent of higher bin (high-low)/high
    Output: tuple of bins, midpoints with the same interpretation as the inputs of the same name
    '''
    # remove all bins and midpoints corresponding to bins
    # where the midpoint (ie rounding point)
    # is greater than the highest tolerated midpoint
    high_bins = midpoints >= high
    bins = bins[~high_bins]
    midpoints = midpoints[~high_bins]
    del high_bins
    # remove all bins and midpoints corresponding to bins
    # where the midpoitn (ie rounding point) is lower than the lowest
    # tolerated rounding point
    low_bins = midpoints <= low
    bins = bins[~low_bins]
    midpoints = midpoints[~low_bins]

    # get a boolean array giving whether the difference between each midpoint[i]
    # and midpoint[i+1] is less than the tolerated difference
    diff = get_diff(midpoints, abs_tol, perc_tol)
    # while at least one midpoint is less than the difference
    while diff.any():
        # grab the index of the lower midpoint in the comparison
        left_ind = np.where(diff)[0][0]

        #############################################################
        # Deprecated
        # right_ind = left_ind + 1
        # midpoint = (midpoints[left_ind] + midpoints[right_ind]) / 2
        # midpoints.delete([left_ind, right_ind])
        # midpoints.insert(midpoint, left_ind)
        ############################################################

        # delete this midpoint from the midpoint array
        midpoints = np.delete(midpoints, [left_ind])
        # delete the corresponding right side of the lower bin
        # from the bins array
        bins = np.delete(bins, [left_ind])
        if left_ind != 0:
            # if the lower midpoint in the comparison is not the lowest
            # midpoint in the arary, recalculate the left side
            # denoting the right edge of the bin beneath
            bins[left_ind - 1] = (midpoints[left_ind - 1] +
                                  midpoints[left_ind]) / 2
        diff = get_diff(midpoints, abs_tol, perc_tol)
    return bins, midpoints


def get_turn_desc(turn):
    '''
    Description: Extracts turn type and turn round number 
    from turn name where possible. If the initial offering
    is the last observed offer (thread has not started), 
    we return a tuple where both elements are None
    Input: 
        turn: string giving turn name
    Output: tuple of length 2 where the first element is a
    string giving the type of turn 's' or 'b' and the second
    element is an integer giving the number of the turn
    '''
    if turn == 'start_price_usd':
        return None, None
    elif len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    else:
        turn_num = turn[1]
        turn_type = turn[0]
        turn_num = int(turn_num)
        return turn_type, turn_num


def dig_norm(df, sig_digs, offr_name):
    '''
    Description: Rounds the offr_name column in df to sig_digs number
    of decimals
    Input:
        df: pd.DataFrame containing offr_name column
        sig_digs: number of decimals to retain
        offr_name: name of the offer in df that will be rounded
    Output: df containing rounded column
    '''
    # grab offer values
    print(offr_name)
    offr = df[offr_name].values
    # some error checking -- values inputted should be in range [0,1]
    off_max = np.amax(offr)
    off_min = np.amin(offr)
    if off_max > 1:
        raise ValueError(
            'Offr max should be 1 or less than 1, Actual max is: %.2f' % off_max)
    if off_min < 0:
        raise ValueError(
            'Offr min should be 0 or greater than 0, Actual min is %.2f' % off_min)
    # grab corresponding index
    ind = df[offr_name].index
    # round offers
    offr = np.around(offr, sig_digs)
    # place rounded offers back in the same column
    df[offr_name] = pd.Series(offr, index=ind)
    return df


def get_round_vals(sig_digs):
    '''
    Description: Creates a list of values from [0,1] inclusive
    equally spaced at increments of 10^(-sig_digs). These 
    are intended to mimick the class values generated 
    by bins_from_common and bins_from_even. Intended to be used
    as classes in downstream transition probability after 
    rounding offer values to sig_digs decimal places 
    and chopping values greater than 1 or less than 0
    Input: Number of decimals to be rounded to
    Output: List of rounded values used downstream as classes
    '''
    init = 0
    last = 1
    step = math.pow(10, -1 * sig_digs)
    midpoints = np.arange(init, last + step, step)
    midpoints = np.around(midpoints, sig_digs)
    return midpoints


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', action='store', type=str)
    # gives the turn immediately previous to the turn this data set
    # is used to predict (ie the last observed turn)
    parser.add_argument('--turn', action='store', type=str)
    # gives the name of the current experiment
    parser.add_argument('--exp', action='store', type=str)
    # gives the number of decimals that should be rounded to
    # for normalized offers
    parser.add_argument('--sig', action='store', type=int)

    # parse args
    args = parser.parse_args()
    name = args.name
    filename = args.name + '_concat.csv'
    turn = args.turn.strip()
    exp_name = args.exp
    sig_digs = args.sig

    # load data frame
    print(turn)
    print(filename)
    df = pd.read_csv('data/exps/%s/%s/%s' %
                     (exp_name, turn, filename), index_col=False)

    # drop abhorent columns as necessary
    if 'unique_thread_id' in df.columns:
        df.drop(columns=['unique_thread_id'], inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    # get response turn
    turn_type, turn_num = get_turn_desc(turn)

    # ensure we iterate over times only when there times to iterate over
    if turn != 'start_price_usd':
        # drop all threads with times outside the range 0-48*60
        count = 0
        tot = len(df.index)
        # iterate over turn numbers through the current turn number
        for i in range(turn_num + 1):
            # grab buyer times if i not 0 b/c time_b0 doesn't exist
            if i != 0:
                low_b = df[df['time_b%d' % i] < 0].index
                high_b = df[df['time_b%d' % i] > 48*60].index
            else:
                low_b = None
                high_b = None
            # grab out of bounds ids for the current seller turn
            low_s = df[df['time_s%d' % i] < 0].index
            high_s = df[df['time_s%d' % i] > 48*60].index
            threads = np.unique(np.append(low_s.values, high_s.values))
            # ensure that low and high b exist before trying to append
            # them to the thread id list
            if low_b is not None or high_b is not None:
                threads = np.unique(np.append(threads, low_b.values))
                threads = np.unique(np.append(threads, high_b.values))
            # grab buyer turn for the next turn if the last observed
            # turn in this data set is a seller turn
            if turn_type == 's' and i == turn_num:
                low_b = df[df['time_b%d' % i] < 0].index
                high_b = df[df['time_b%d' % i] > 48*60].index
                threads = np.unique(np.append(threads, low_b))
                threads = np.unique(np.append(threads, high_b))
            # increment count of dropped threads
            count = count + threads.size
            # actually drop threads from data frame
            df.drop(index=threads, inplace=True)

        print('Dropped %.2f %% of threads' % (count / tot))

    # grab bins, using midpoint step algorithm if step flag is given
    if name == 'train' or name == 'toy':
        bins = None
        midpoints = get_round_vals(sig_digs)
        # NOTE Improve by not hardcoding later if necessary
        time_bins, time_midpoints = bin_times_from_midpoints(15)
        # for the training data
        if name == 'train':
            # pickle offer bins to use in training and binning test data
            pic_dic = {'bins': bins, 'midpoints': midpoints}
            bins_pick = open('data/exps/%s/%s/bins.pickle' %
                             (exp_name, turn), 'wb')
            pickle.dump(pic_dic, bins_pick)
            bins_pick.close()

            # pickle time data binning arrays
            if turn != 'start_price_usd':
                pic_dic = {'time_bins': time_bins,
                           'time_midpoints': time_midpoints}
                bins_pick = open('data/exps/%s/%s/time_bins.pickle' %
                                 (exp_name, turn), 'wb')
                pickle.dump(pic_dic, bins_pick)
                bins_pick.close()

    elif name == 'test':
        # load offer and time bins from corresponding pickles
        f = open("data/exps/%s/%s/bins.pickle" % (exp_name, turn), "rb")
        pic_dic = pickle.load(f)
        bins = pic_dic['bins']
        midpoints = pic_dic['midpoints']
        f.close()
        # only load time bins if turn is not start_price_usd:
        if turn != 'start_price_usd':
            f = open("data/exps/%s/%s/time_bins.pickle" %
                     (exp_name, turn), "rb")
            pic_dic = pickle.load(f)
            time_bins = pic_dic['time_bins']
            time_midpoints = pic_dic['time_midpoints']
            f.close()
        del pic_dic

    if turn != 'start_price_usd':
        # iterate over all offr_ji features in the data set
        for i in range(turn_num + 1):
            # bin all seller / buyer offers for the current turns in the
            # bins established above
            df = dig_norm(df, sig_digs, 'offr_s' + str(i))
            df = dig_norm(df, sig_digs, 'offr_b' + str(i))

            # bin buyer times except for b0
            if i != 0:
                df = digitize(df, time_bins, time_midpoints, 'time_b%d' % i)
            # bin seller times for each observed turn
            df = digitize(df, time_bins, time_midpoints, 'time_s%d' % i)

            if turn_type == 's' and i == turn_num:
                df = dig_norm(df, sig_digs, 'offr_b' + str(i + 1))
                df = digitize(df, time_bins, time_midpoints,
                              'time_b' + str(i + 1))
    # if the last observed turn is the initial offering, there are no
    # times to digitize and only one offer -- b0
    else:
        df = digitize(df, bins, midpoints, 'offr_b0')
    # saves the resulting data frame after manipulations
    df.to_csv('data/exps/%s/binned/%s_%s.csv' %
              (exp_name, filename.replace('.csv', ''), turn), index_label=False)


if __name__ == '__main__':
    main()
