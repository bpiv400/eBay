"""
Generates a dataframe of time-valued features
for at level of slr, item, lstg

"""
import argparse
import pandas as pd
import numpy as np
from util_env import extract_datatype
from time_funcs import *

# dictionary of listing level features
LSTG_FEATS = {
    'lstg_max': (lstg_max, 0)
}
# dictionary of seller level features
SLR_FEATS = {
    'slr_open': (slr_open, 0)
}
# dictionary of item level features
ITEM_FEATS = {
    'item_open': (item_open, 0)
}


def make_level_feats(offrs, timedf, inds, curr_feats, lstg=False):
    prev = None
    # iterate over indices
    for ix in inds:
        row = timedf.loc[ix, :].squeeze()
        # extract corresponding offer if there is one
        if row['offer']:
            curr_offr = offrs.loc[(
                row['thread'], row['index']), :].squeeze()
        else:
            curr_offr = None
        edges = {
            'start_lstg': row['start_lstg'],
            'end_lstg': row['end_lstg'],
            'start_thread': row['start_thread'],
            'end_thread': row['end_thread']
        }
        # iterate over features
        for feat_name, feat_tuple in curr_feats.items():
            timedf.loc[ix, feat_name] = feat_tuple[0](
                offer=curr_offr, prev_feats=prev, edges=edges)
        prev = timedf.loc[ix, :].squeeze()


def fix_lstgs(timedf, inds, feats):
    feats = feats.keys()
    timedf = timedf.loc[inds]
    end_lstg = timedf.loc[timedf['end_lstg'] == 1, ['lstg']]
    # dropping index so we can max over it directly
    end_lstg.index.name = 'org_ind'
    lst_ind = end_lstg.reset_index(drop=False)
    # find the max index for each lstg
    lst_ind = lst_ind.groupby('lstg').max()['org_ind']
    # grab newly created features for last index
    lst_feats = timedf.loc[lst_ind.values, (list(feats) + ['lstg'])]
    # make original index in timedf a column in end_lstg so it's not lost in merge
    end_lstg.reset_index(drop=False, inplace=True)
    # merge end lstg with lst feats to create a dataframe with 1 row for each original
    # index where each matches with the corresponding lstg row in lst_feats containing
    # new features
    lst_feats = pd.merge(end_lstg, lst_feats, on='lstg', how='inner')
    # reset the index to be the same as in the original dataframe
    lst_feats.set_index('org_ind', drop=True, inplace=True)
    # drop the lstg column
    lst_feats.drop(columns='lstg', inplace=True)
    # update timedf slice
    timedf.loc[lst_feats.index, lst_feats.columns.values] = lst_feats
    return timedf


def fix_end_lstg(timedf):
    end_lstg = timedf.loc[timedf['end_lstg'] == 1, 'lstg']
    # dropping index so we can max over it directly
    end_lstg.index.name = 'org_ind'
    lst_ind = end_lstg.reset_index(drop=False)
    # find the max index for each lstg
    lst_ind = lst_ind.groupby('lstg').max()['org_ind']
    # reset all to 0
    timedf.loc[end_lstg.index, 'end_lstg'] = 0
    # set last in each lstg to 1
    timedf.loc[lst_ind.values, 'end_lstg'] = 1
    return timedf


def add_time_feats(offrs, timedf, slr_feats, item_feats, lstg_feats):
    """
    Groups offrs, threads, lstgs, time_feats by item and
    computes particular time-valued features for each
    """
    # set new index for offer dataframe
    offrs.set_index(['thread', 'index'], inplace=True, drop=False)
    # sort the entire fucking mess by offer time
    timedf.sort_values('clock', ascending=True, inplace=True, kind='quicksort')
    fix_end_lstg(timedf)
    assert (timedf.groupby('lstg').sum()['end_lstg'] == 1).all()
    assert (timedf.groupby('lstg').sum()['start_lstg'] == 1).all()
    # iterate over feature dictionaries
    for feat_dict in [SLR_FEATS, LSTG_FEATS, ITEM_FEATS]:
        # iterate over contents of current dictionary
        for feat_name, feat_tuple in feat_dict.items():
            timedf[feat_name] = feat_tuple[1]
    # split dataframe into groups of sellers
    slr_groups = timedf.groupby('slr')
    for slr, slr_inds in slr_groups.groups.items():
        # split current dataframe into groups of items
        curr_slr = timedf.loc[slr_inds, :]
        for item, item_inds in curr_slr.groupby(['title', 'cndtn']).groups.items():
            # subset the dataframe to the current item
            curr_item = timedf.loc[item_inds, :]
            # iterate over listings in the current item
            for lstg, lstg_inds in curr_item.groupby('lstg').groups.items():
                # compute level features
                make_level_feats(offrs, timedf, lstg_inds, lstg_feats)
            # fix listing end timelag for lstg level features
            timedf.loc[item_inds] = fix_lstgs(timedf, item_inds, lstg_feats)
            # make item level features
            make_level_feats(offrs, timedf, item_inds, item_feats)
        # fix listing timelag for item level features
        timedf.loc[item_inds] = fix_lstgs(timedf, slr_inds, item_feats)
        # make seller level features
        make_level_feats(offrs, timedf, slr_inds, slr_feats)
    # updating all end of lstg indices to last lstg value
    timedf.loc[item_inds] = fix_lstgs(timedf, slr_inds, item_feats)
    # timedf.loc[timedf['slr']]
    print(timedf.loc[timedf['slr'].isin(
        [28260, 24330, 8130, 5797, 13386]), ['clock', 'lstg', 'start_lstg', 'end_lstg', 'slr_open', 'item_open']])
    return timedf


def tighten_removes(offrs, time_feats):
    """
    Since end date for each listing is given in days but offers
    are given in seconds, there will be listings which 'end' after they've
    been sold

    We remove the day-denominated end entries for those listings
    and update end_ind as appropriate to the last accepted offers
    """
    # NOTE: temporary fix for multiple sales problem
    offrs = offrs.loc[offrs['lstg'] != 44636280, :]
    # set index of offrs to listing, index
    # subset offers to only include the last offer in each thread
    last_offrs = offrs.groupby('thread').max()['index']
    last_offrs = pd.DataFrame(
        {'last_offr': last_offrs.to_numpy()}, index=last_offrs.index)
    last_offrs = last_offrs.loc[last_offrs['last_offr'] > 1, :]
    last_offrs.index.name = 'thread'
    offrs = offrs.copy().join(last_offrs, on='thread', how='inner')
    last_offrs = offrs.loc[offrs['index'] == offrs['last_offr'], :]
    prev_offrs = offrs.loc[offrs['index'] == (offrs['last_offr'] - 1), :]
    last_offrs.set_index('thread', drop=True, inplace=True)
    prev_offrs.set_index('thread', drop=True, inplace=True)
    last_offrs = last_offrs.loc[last_offrs['price'] == prev_offrs['price'], :]
    last_offrs.reset_index(inplace=True, drop=False)
    # guarantee that each listing has at most 1 acceptance
    assert len(last_offrs) == len(last_offrs['lstg'].unique())
    # corresponding listings in time feats
    end_times = time_feats.loc[time_feats['lstg'].isin(
        last_offrs['lstg']) & time_feats['end_lstg'] == 1, ['lstg', 'clock']]
    # join frame of end times with frame of accepted offer times
    end_times.index.name = 'time_index'
    end_times.rename(columns={'clock': 'end_time'}, inplace=True)
    end_times = pd.merge(end_times, last_offrs, on='lstg', how='inner')
    # find indices in time_feats where the end time occurs after the accepted offer
    end_times = end_times[end_times['end_time'] > end_times['clock']]
    end_times.rename(columns={'clock': 'true_end'}, inplace=True)
    # for each listing, find the index of the last offer
    time_feats = pd.merge(
        time_feats, end_times[['lstg', 'end_time', 'true_end']], how='left', on='lstg')
    time_feats.loc[time_feats['true_end'] ==
                   time_feats['clock'], 'end_lstg'] = 1
    time_feats.drop(index=time_feats[time_feats['clock']
                                     == time_feats['end_time']].index, inplace=True)
    time_feats.drop(columns=['end_time', 'true_end'], inplace=True)
    return time_feats


def gen_timedf(offrs, threads, listings):
    """"
    Creates a dataframe for storing time valued features later
    Columns include:
        -lstg
        -slr
        -clock (in seconds)
        -start_ind
        -end_ind
    One row is given for the start day of each listing and then each
    ensuing offer

    Assumes start and end are days from 1/1/1960 (exclusive)

    Returns
        Tuple of:
        -offer dataframe joined with other inputs to include
        slr, lstg, item, start_price as well original features and
        -time_feats df
    """
    threads.reset_index(drop=False, inplace=True)
    listings.reset_index(
        drop=False, inplace=True)  # remove lstg as listing index
    # grab all start days from listings
    time_feats = listings.loc[:, [
        'lstg', 'start_date', 'slr', 'title', 'cndtn']]
    # add indicator to convey these are start days
    time_feats.rename(columns={'start_date': 'clock'}, inplace=True)
    time_feats.loc[:, 'start_lstg'] = 1
    time_feats.loc[:, 'end_lstg'] = 0
    # converts clock to seconds
    time_feats.loc[:, 'clock'] = 60 * 60 * 24 * time_feats.loc[:, 'clock']
    # grab end days
    ends = listings.loc[:, ['lstg', 'end_date', 'slr', 'title', 'cndtn']]
    ends.rename(columns={'end_date': 'clock'}, inplace=True)
    ends.loc[:, 'start_lstg'] = 0
    ends.loc[:, 'end_lstg'] = 1
    # convert clock to seconds (pushes end time to the end of the current day)
    ends.loc[:, 'clock'] = 60 * 60 * 24 * (ends.loc[:, 'clock'] + 1)
    # append ends to starts
    time_feats = time_feats.append(ends)
    time_feats['thread'] = np.NaN
    time_feats['index'] = np.NaN
    time_feats['start_thread'] = 0
    time_feats['end_thread'] = 0
    print("Join with threads")
    listings.drop(columns='byr_us', inplace=True)
    # join listing file with threads file, storing listing index, thread index, and start price
    threads = pd.merge(listings, threads, on='lstg', how='inner')
    # define offers output
    offrs.reset_index(drop=False, inplace=True)
    # join threads with offrs
    print(offrs.columns)
    print(threads.columns)
    offrs = pd.merge(offrs, threads, on='thread', how='inner')
    # add feature denoting an offer ends a thread
    last_offrs = offrs.groupby('thread').max()['index']
    last_offrs = pd.DataFrame(
        {'last_offr': last_offrs.to_numpy()}, index=last_offrs.index)
    last_offrs.index.name = 'thread'
    offrs = offrs.join(last_offrs, how='inner', on='thread')
    offrs['end_thread'] = 0
    offrs.loc[offrs['last_offr'] == offrs['index'], 'end_thread'] = 1
    offrs.drop(columns='last_offr', inplace=True)
    offrs['start_thread'] = 0
    offrs.loc[offrs['index'] == 1, 'start_thread'] = 1
    offrs.loc[:, 'clock'] = offrs.loc[:, 'start_time'] + offrs.loc[:, 'clock']
    offr_times = offrs.loc[:, ['clock', 'slr', 'start_thread', 'end_thread',
                               'lstg', 'cndtn', 'thread', 'index', 'title']]
    offr_times['offer'] = 1
    time_feats.loc[:, 'offer'] = 0
    offr_times.loc[:, 'end_lstg'] = 0
    offr_times.loc[:, 'start_lstg'] = 0
    # add features for start and end of a thread
    # append to time valued features df
    time_feats = time_feats.append(offr_times)
    time_feats.reset_index(drop=True, inplace=True)
    return time_feats, offrs


def get_time_feats(offrs, threads, lstgs, lstg_feats, item_feats, slr_feats):
    print("Generating time df")
    timedf, offrs = gen_timedf(offrs.copy(), threads.copy(), lstgs.copy())
    timedf = tighten_removes(offrs, timedf)
    print(timedf.groupby('slr').count()['lstg'].sort_values())
    print(offrs.columns)
    timedf = add_time_feats(offrs, timedf, slr_feats, item_feats, lstg_feats)
    return timedf
    raise ValueError("Done")
