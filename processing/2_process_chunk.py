"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from time_funcs import *

END = 136079999
DIR = '../../data/chunks/'
LEVELS = ['slr', 'meta', 'leaf', 'title', 'cndtn', 'lstg']
FUNCTIONS = {'slr': [open_lstgs, open_threads],
             'meta': [open_lstgs, open_threads],
             'leaf': [open_lstgs, open_threads],
             'title': [open_lstgs, open_threads],
             'cndtn': [slr_min, byr_max,
                       slr_offers, byr_offers,
                       open_lstgs, open_threads],
             'lstg': [slr_min, byr_max,
                      slr_offers, byr_offers,
                      open_threads]}


def create_obs(listings, cols, keep=None):
    df = listings[cols].copy()
    if keep is not None:
        df = df.loc[keep]
    df = df.rename(axis=1,
                   mapper={'start_price': 'price', 'start': 'clock', 'end': 'clock'})
    df['index'] = int('end' in cols)
    if 'start_price' not in cols:
        df['price'] = -1
    df['accept'] = int('end' in cols and 'start_price' in cols)
    df['reject'] = (df['price'] == -1).astype(np.int64)
    df.set_index('index', append=True, inplace=True)
    return df


def add_start_end(offers, listings):
    # create data frames to append to offers
    start = create_obs(listings, ['start', 'start_price'])
    bins = create_obs(listings, ['end', 'start_price'],
                      listings['bo'] == 0)
    end = create_obs(listings, ['end'],
                     (listings['bo'] != 0) & (listings['accept'] != 1))
    # append to offers
    offers = offers.append(start).append(bins).append(end)
    # put clock into index
    offers.set_index('clock', append=True, inplace=True)
    # sort
    offers.sort_values(offers.index.names, inplace=True)
    return offers


def reindex(df):
    df.set_index(LEVELS[:5], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_listings(L, offers):
    listings = L[LEVELS[:5] + ['start_date',
                               'end_date', 'start_price', 'bo']].copy()
    listings['thread'] = 0
    listings.set_index('thread', append=True, inplace=True)
    listings = reindex(listings)
    # convert start and end dates to seconds
    listings['end_date'] += 1
    for z in ['start', 'end']:
        listings[z + '_date'] *= 60 * 60 * 24
        listings.rename(columns={z + '_date': z}, inplace=True)
    listings['end'] -= 1
    # indicator for whether a bo was accepted
    listings = listings.join(offers['accept'].groupby('lstg').max())
    # change end time if bo was accepted
    t_accept = offers.loc[offers['accept'] == 1, 'clock']
    listings = listings.join(t_accept.groupby('lstg').min())
    listings['end'] = listings[['end', 'clock']].min(axis=1).astype(np.int64)
    listings.drop('clock', axis=1, inplace=True)
    return listings


def init_offers(L, T, O):
    offers = O[['clock', 'price', 'accept', 'reject']].copy()
    offers = offers.join(T['start_time'])
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:5]])
    offers = reindex(offers)
    return offers


def get_offers(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # listings data frame
    listings = init_listings(L, offers)
    # add buy it nows
    offers = add_start_end(offers, listings)
    # start price for normalization
    start_price = listings['start_price'].reset_index(
        level='thread', drop=True)
    return offers, start_price


def get_continuous_df(offers):
    """
    Build dataframe of index:
    slr meta leaf  title    cndtn lstg, 
    where each row gives the start time ('start') and 
    end time ('end') of that listing and 'accept' gives
    whether the listing was sold
    """
    # subset to only thread 0 and last thread for each lstg
    zeros = offers.loc[offers['thread' == 0], :]
    zeros.reset_index(level='clock', drop=False, inplace=True)
    zeros.reset_index(level='thread', drop=False, inplace=True)
    # lstgs that end in an acceptance
    accepts = offers.loc[offers['accept' == 1]]  # accept indices
    accepts.reset_index(level=['thread', 'index'], drop=True, inplace=True)
    accepts.reset_index(level='clock', drop=False, inplace=True)
    accepts.drop(columns=['price', 'accept', 'reject'], inplace=True)
    unique_lstg = len(accepts.get_level_values('lstg').unique())
    # ensure each lstg has at most 1 acceptance
    assert unique_lstg == len(accepts)
    accepts['index'] = 1
    accepts['offer_accept'] = 1
    accepts.set_index('index', inplace=True, drop=True, append=True)
    accepts.index = accepts.index.reorder_levels(list(zeros.index.names))
    accepts.rename({'clock': 'accept_clock'}, inplace=True)
    lstgs = pd.merge(zeros, accepts, how='left',
                     left_index=True, right_index=True)
    accepts = lstgs['offer_accept'] == 1
    lstgs.loc[accepts, 'clock'] = lstgs.loc[accepts, 'accept_clock']
    lstgs.drop(columns=['accept_clock',
                        'offer_accept', 'reject'], inplace=True)
    # ensure lstgs contains two rows for each lstg (1 for start, 1 for end)
    assert len(lstgs) == 2 * len(lstgs.index.get_level_values('lstg').unqiue())
    # split lstg into start rows and end rows
    starts = lstgs.xs(0, level='index', drop=True)
    starts.drop(columns='accept', inplace=True)
    starts.rename({'clock': 'start'}, inplace=True)
    ends = lstgs.xs(1, level='index', drop=True)
    ends.rename({'clock': 'end'}, inplace=True)
    # ensure that each lstg has a start and end
    assert len(starts) == starts.index.get_level_values('lstg').unqiue()
    assert len(ends) == ends.index.get_level_values('lstg').unqiue()
    assert len(starts) == len(ends)
    assert len(ends) == 1 / 2 * len(lstgs)
    # merge starts and ends
    lstgs = pd.merge(starts, ends, how='inner',
                     right_index=True, left_index=True)
    return lstgs


def get_cont_index(lstgs):
    """
    Replace unique cont-item pairs with unique const index
    """
    mapping = lstgs.reset_index(drop=False)
    mapping = mapping.loc[:, ['item', 'cont']]
    mapping['def'] = True
    mapping.drop_duplicates(inplace=True, keep='first')
    mapping.reset_index(inplace=True, drop=True)
    mapping.reset_index(inplace=True, drop=False)
    mapping.rename(columns={'index': 'true_cont'}, inplace=True)
    mapping.drop(columns='def', inplace=True)
    lstgs.reset_index(inplace=True, drop=False)
    lstgs = pd.merge(lstgs, mapping, how='inner', on=['lstg', 'cont'])
    lstgs.drop(columns='cont', inplace=True)
    lstgs.rename(columns={'true_cont': 'cont'}, inplace=True)
    lstgs.drop(columns=['item', 'start', 'end', 'accept'], inplace=True)
    lstgs.set_index('lstg', inplace=True)


def get_item_index(lstgs):
    """
    Replace slr-title-condtn with item identifer in lstgs
    (inplace) and return a mapping dataframe with columns
    slr, title, condtn, item
    """
    mapping = lstgs.reset_index(drop=False)
    mapping = mapping.loc[:, ['slr', 'title', 'condtn']]
    mapping['def'] = True
    mapping.drop_duplicates(inplace=True, keep='first')
    mapping.reset_index(inplace=True, drop=True)
    mapping.reset_index(inplace=True, drop=False)  # create mapper var
    mapping.rename(columns={'index': 'item'}, inplace=True)
    mapping.drop(columns='def', inplace=True)
    lstgs.reset_index(drop=False, inplace=True)
    lstgs = pd.merge(lstgs, mapping, how='inner',
                     on=['slr', 'title', 'condtn'])
    lstgs.drop(columns=['slr', 'title', 'condtn'], inplace=True)
    return mapping


def broadcast_items(targdf, sourcedf, tcol, scol):
    """
    Broadcasts values of sourcedf through items in targdf
    """
    sourcedf = sourcedf.reindex(index=targdf.index, level='item')
    invalids = sourcedf[sourcedf[scol].isna()].index
    sourcedf.drop(index=invalids, inplace=True)
    targdf.loc[sourcedf.index, tcol] = sourcedf[scol]
    pass


def get_candidate_lstgs(cands):
    """
    Returns rows of dataframe corresponding to the listing in each item,
    for which the finish time of the current listing occurs before
    the start time of the earliest added to each listing sequence
    """
    cands['util'] = cands['end'] < cands['early_start']
    firsts = cands.groupby('item').cumsum()
    firsts = cands.loc[firsts['util'] == 1, :]
    return firsts


def longest(lstgs, unsold, counter, shadow=None):
    """
    Finds the longest non-overlapping sequence of intervals
    among listings in unsold dataframe. Counter denotes the
    number of 'cont' in lstgs corresponding to the current
    sequences being built for each item

    Shadow is not null if unsold is a copy of the actual unsold
    dataframe which must be subset (i.e. in the case where we have
    subset the dataframe to only listings that began before the selling
    of the current listing occurred)
    """
    cands = get_candidate_lstgs(unsold)
    while len(cands) > 0:
        # add selected listings to the current sequences being built
        lstgs.loc[cands.index, 'cont'] = counter
        # remove from unsold and unsold subset
        unsold.drop(index=cands.index, inplace=True)
        if shadow is not None:
            shadow.drop(index=cands.index, inplace=True)
        # update value of early_start for each item
        cands.index = cands.index.droplevel('lstg')
        broadcast_items(unsold, cands, 'early_start', 'start')
        # update candidates using new list of unsold listings
        cands = get_candidate_lstgs(unsold)
    pass


def add_continuous_lstg(offers):
    """
    Adds an index variable for continuous listings
    In the dataset, we have items (e.g. slr-condtn-title tuples)
    which correspond to the same item. If these appear simultaneously,
    i.e. if the listing with the earlier finish time finishes after
    the other starts
    """
    print("adding continuous listing")
    lstgs = get_continuous_df(offers)
    # set index to slr-title-condtn-lstg
    lstgs.drop(level=['meta', 'leaf'], inplace=True)
    lstgs.index = lstgs.index.reorder_levels(
        ['slr', 'title', 'condtn', 'lstg'])
    # make slr-title-condtn -> item mapping and reset
    # lstgs index to 'item', 'lstg'
    mapping = get_item_index(lstgs)
    lstgs['cont'] = 0
    sold = lstgs.loc[lstgs['accept'] == 1, ['start', 'end']]
    unsold = lstgs.loc[lstgs['accept'] == 0, ['start', 'end']]
    # sort sold by ascending finish time
    sold.sort(columns='end', ascending=True, inplace=True)
    # sort unsold by descending start time
    unsold.sort(columns='start', ascending=False, inplace=True)
    # add utility column to sold
    sold['util'] == True
    # initialize sequence counter
    counter = 1
    while len(sold) > 0:
        # indicate that the most recent sold copy for each item never sold
        unsold['sold_finish'] = np.inf
        unsold['early_start'] = np.inf
        # extract indices of sold lstgs with earliest finish time for each item
        order = sold.groupby(level=['item']).cumsum()['util']
        firsts = unsold[order == 1]
        # add to sequence counter for each lstg
        lstgs.loc[firsts.index, 'cont'] = counter
        # remove from sold
        sold.drop(index=firsts.index, inplace=True)
        if len(unsold) > 0:
            # drop lstg from extracted index
            firsts.index = firsts.index.droplevel('lstg')
            # set sold_finish/early_start for unsold lstgs and subset to lstgs
            #  that started before each in-item sale
            broadcast_items(unsold, firsts, 'sold_finish', 'end')
            broadcast_items(unsold, firsts, 'early_start', 'start')
            unsold_subset = unsold.loc[unsold['sold_finish']
                                       > unsold['start'], :]
            longest(lstgs, unsold_subset, counter, shadow=unsold)
        counter += 1  # increment sequence counter
    # after having removed all sold listings, if any unsold listings remain,
    # greedily combine these into non-overlapping sequences of maximum length within
    # each item
    while len(unsold) > 0:
        unsold['early_start'] = np.inf
        longest(lstgs, unsold_subset, counter)
        counter += 1
    # generate a unique index for each cont-item pair in lstg and replace
    # cont with this new value
    get_cont_index(lstgs)
    # ensure all listings are members of some continous listing
    assert (lstgs['cont'] != 0).all()
    offers = pd.merge(offers, lstgs, how='inner',
                      left_on='lstg', left_index=False, right_on='lstg')
    return offers


def get_time_feats(L, T, O):
    print('Creating offer dataframe')
    offers, start_price = get_offers(L, T, O)
    add_continuous_lstg(offers)
    print('Creating time features')
    time_feats = pd.DataFrame(index=offers.index)
    for i in range(len(LEVELS)):
        levels = LEVELS[: i+1]
        ordered = offers.copy().sort_values(levels + ['clock'])
        name = levels[-1]
        f = FUNCTIONS[name]
        for j in range(len(f)):
            feat = f[j](ordered.copy(), levels)
            if f[j].__name__ in ['slr_min', 'byr_max']:
                feat = feat.div(start_price)
                feat = feat.reorder_levels(
                    LEVELS + ['thread', 'index', 'clock'])
            newname = '_'.join([name, f[j].__name__])
            print('\t%s' % newname)
            time_feats[newname] = feat
    # set [lstg, thread, index] as index
    time_feats.reset_index(level=LEVELS[:len(LEVELS)-1],
                           drop=True, inplace=True)
    time_feats.reset_index(level='clock', inplace=True)
    return time_feats


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print("Loading data")
    chunk = pickle.load(open(DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # time-varying features
    time_feats = get_time_feats(L, T, O)

    # ids of threads to keep
    T = T.join(L.drop(['bo', 'title', 'end_date'], axis=1))
    end_time = time_feats['clock'].groupby(level='lstg').max()
    keep = (T.bin_rev == 0) & (T.flag == 0) & (end_time < END)
    T = T.loc[keep].drop(['start_time', 'bin_rev', 'flag'], axis=1)
    O = O.loc[keep][['price', 'message']]

    # write simulator chunk
    print("Writing first chunk")
    chunk = {'O': O,
             'T': T,
             'time_feats': time_feats}
    pickle.dump(chunk, open(DIR + '%d_simulator.pkl' % num, 'wb'))
