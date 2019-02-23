"""
Generates a dataframe of time-valued features
for each item

"""
import argparse
import pandas as pd
import numpy as np
from util_env import extract_datatype


def add_time_feats(offrs, threads, lstgs, time_feats):
    """
    Groups offrs, threads, lstgs, time_feats by item and
    computes particular time-valued features for each
    """
    time_feats.groupby('item')


def tighten_removes(offrs, time_feats):
    """
    Since end date for each listing is given in days but offers
    are given in seconds, there will be listings which 'end' after they've
    been sold

    We remove the day-denominated end entries for those listings
    and update end_ind as appropriate to the last accepted offers
    """
    # set index of offrs to listing, index
    # subset offers to only include the last offer in each thread
    last_offrs = offrs.groupby('thread').max('index')
    last_offrs = pd.DataFrame(
        {'last_offr': last_offrs.data}, index=last_offrs.index)
    last_offrs.index.name = 'thread'
    offrs = offrs.join(last_offrs, on='thread')
    last_offrs = offrs.loc[offrs['index'] == offrs['last_offr'], :]
    # subset these to only include acceptances
    last_offrs = last_offrs.loc[last_offrs['type'] == 2, :]
    # guarantee that each listing has at most 1 acceptance
    assert len(last_offrs) == len(last_offrs['lstg'].unique())
    # corresponding listings in time feats
    end_times = time_feats.loc[time_feats['lstg'].isin(
        last_offrs['lstg']) and time_feats['end_ind'] == 1, ['lstg', 'clock']]
    # join frame of end times with frame of accepted offer times
    end_times.index.name = 'time_index'
    end_times.rename(columns={'end_time': 'clock'}, inplace=True)
    end_times = end_times.join(last_offrs, on='lstg', how='inner')
    # find indices in time_feats where the end time occurs after the accepted offer
    end_times = end_times[end_times['end_time'] > end_times['clock']]
    end_lstgs = end_times['lstg']
    end_times = end_times.index
    # drop these indices from time features
    time_feats.drop(index=end_times, inplace=True)
    # for each listing, find the index of the last offer
    last_offrs = time_feats.groupby('lstg').idxmax('clock')
    # subset to only those with lstg close removed
    last_offrs = last_offrs.loc[end_lstgs.values]
    # set end_ind on these new last offers
    time_feats.loc[last_offrs['clock'].values, 'end_ind'] = 1
    return time_feats


def gen_timedf(offrs, threads, listings):
    """"
    Creates a dataframe for storing time valued features later
    Columns include:
        -item
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
    # grab all start days from listings
    time_feats = listings.loc[:, ['item', 'lstg', 'start', 'slr']]
    # add indicator to convey these are start days
    time_feats.rename(columns={'start': 'clock'}, inplace=True)
    time_feats.loc[:, 'start_ind'] = 1
    time_feats.loc[:, 'end_ind'] = 0
    # converts clock to seconds
    time_feats.loc[:, 'clock'] = 60 * 60 * 24 * time_feats.loc[:, 'clock']
    # grab end days
    ends = listings.loc[:, ['item', 'lstg', 'end', 'slr']]
    ends.rename(columns={'clock': 'end'}, inplace=True)
    ends.loc[:, 'start_ind'] = 0
    ends.loc[:, 'end_ind'] = 1
    # convert clock to seconds (pushes end time to the end of the current day)
    ends.loc[:, 'clock'] = 60 * 60 * 24 * (ends.loc[:, 'clock'] + 1)
    # append ends to starts
    time_feats = time_feats.append(ends)
    # join listing file with threads file, storing listing index, thread index, and start price
    threads = threads.join(listings, on='lstg')
    # remove extra cols
    threads = threads.loc[:, ['thread', 'lstg', 'item', 'slr', 'start_price']]
    # join threads with offrs
    offrs = offrs.join(threads, on='thread')
    offr_times = offrs.loc[:, ['item', 'clock', 'slr', 'lstg']]
    offr_times.loc[:, ['end_ind', 'start_ind']] = 0
    # append to time valued features df
    time_feats = time_feats.append(offr_times)
    return time_feats, offrs


def get_time_feats(offrs, threads, lstgs):
    # create dummy time-valued feature
    time_stamps = [6*60 ^ 2*364*24, 6*60 ^ 2*367*24, 6*60 ^ 2*369*24]
    time_index = pd.MultiIndex.from_product(
        [lstgs['item'].unique(), time_stamps])
    time_df = pd.DataFrame(0, columns=['a'], index=time_index)
    return time_df
    # timedf, offrs = gen_timedf(offrs, threads, lstgs)
    # timedf = tighten_removes(offrs, timedf)
