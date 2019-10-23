from datetime import datetime as dt
import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import math
import gc

def date_feats(feat_df):
    # grab offer time
    off_series = feat_df['src_cre_date'].values
    # grab auction post time
    post_series = feat_df['auct_start_dt'].values
    # grab auction expiration time
    close_series = feat_df['auct_end_dt'].values
    close_series = close_series + np.timedelta64(24, 'h')

    # get total duration in hours
    dur = (close_series - post_series).astype(int)/1e9/math.pow(60, 2)

    rem = (close_series - off_series).astype(int)/1e9/math.pow(60, 2)
    passed = (off_series - post_series).astype(int)/1e9/math.pow(60, 2)

    # creating series for each new feature
    duration = pd.Series(dur, index=feat_df.index)
    remain = pd.Series(rem, index=feat_df.index)
    passed_time = pd.Series(passed, index=feat_df.index)
    frac_passed = pd.Series(passed/dur, index=feat_df.index)
    frac_remain = pd.Series(remain/dur, index=feat_df.index)

    feat_df['frac_remain'] = frac_remain
    feat_df['frac_passed'] = frac_passed
    feat_df['passed'] = passed_time
    feat_df['remain'] = remain
    feat_df['duration'] = duration

    return feat_df


# def date_feats(feat_df):
#     # grab offer time
#     off_series = feat_df['src_cre_date'].values
#     # grab auction post time
#     post_series = feat_df['auct_start_dt'].values
#     # grab auction expiration time
#     close_series = feat_df['auct_end_dt'].values
#     close_series = close_series + np.timedelta(24, 'h')
#
#     # convert all to dates
#     post_series = pd.to_datetime(post_series).values.astype('datetime64[D]')
#     close_series = pd.to_datetime(close_series).values.astype('datetime64[D]')
#     off_series = pd.to_datetime(off_series).values.astype('datetime64[D]')
#
#     # find the duration of the auction (inclusive)
#     # (expiration time - post time)
#     dur = np.busday_count(post_series, close_series,
#                           weekmask=[1, 1, 1, 1, 1, 1, 1]) + 1
#
#     # find time until the auction closes, remaining time,\
#     #  (expiration time - offer time)
#     rem = np.busday_count(off_series, close_series,
#                           weekmask=[1, 1, 1, 1, 1, 1, 1])
#
#     # convert time until auction close to fraction of
#     # total duration (remaining time / duration)
#     rem = np.divide(rem, dur)
#
#     feat_df['dur_time'] = dur
#     feat_df['rem_time'] = rem
#     return feat_df


def norm_price(feat_df):
    feat_df['norm_offr_price'] = np.divide(
        feat_df['offr_price'], feat_df['start_price_usd'])
    feat_df['norm_rsp_price'] = np.divide(
        feat_df['rsp_offr'], feat_df['start_price_usd'])
    return feat_df

########################################################
# DEPRECATED
# def accept_bool(df):
#     df.sort_values(by='src_cre_date', ascending=True,
#                    inplace=True)
#     accepted = df['status_id'].isin([1, 9]).values
#     tot = np.sum(accepted)
#     if tot > 0:
#         if tot > 1:
#             return False
#         else:
#             return accepted[len(accepted) - 1]
#     else:
#         return True


def remove_accept(df, accept_series, both_inds, val):
    accept_series.drop(index=both_inds, inplace=True)
    big_inds = accept_series[accept_series > 1].index
    big_inds = big_inds.values
    if big_inds.size > 0:
        print(big_inds)
        df_inds = df[df['unique_thread_id'].isin(big_inds)].index
        df.drop(index=df_inds, inplace=True)
    accept_series.drop(index=big_inds, inplace=True)
    remaining_threads = accept_series.index
    remaining_threads = remaining_threads.values
    accepted_df = df.loc[df['unique_thread_id'].isin(remaining_threads), ['status_id', 'unique_thread_id',
                                                                          'turn_count']].copy()
    if len(accepted_df.index) > 0:
        num_turns = accepted_df.groupby('unique_thread_id').size()
        accepted_df = accepted_df[accepted_df['status_id'] == val].copy()
        accepted_df.set_index('unique_thread_id', inplace=True)
        loc_accepted = accepted_df['turn_count']
        num_turns = num_turns - 1
        thread_inds = loc_accepted[num_turns != loc_accepted].index
        thread_inds = thread_inds.values
        print(thread_inds)
        df_inds = df[df['unique_thread_id'].isin(thread_inds)].index
        df.drop(index=df_inds, inplace=True)
    return df


def clean_data(df):
    org_ids = len(np.unique(df['unique_thread_id'].values))
    # remove thread ids corresponding to threads where at least one offer is greater
    # than the start price
    larg_off = df['start_price_usd'].values < df['offr_price'].values
    larg_off_threads = np.unique(df.loc[larg_off, 'unique_thread_id'].values)
    print(larg_off_threads)
    larg_off = df['unique_thread_id'].isin(larg_off_threads)
    del larg_off_threads
    larg_off_inds = df[larg_off].index
    df.drop(larg_off_inds, inplace=True)
    del larg_off
    del larg_off_inds

    print('Removed threads where one offer is greater than the start price')
    # remove thread ids corresponding to threads where more than 6 turns have been taken
    long_thread = df['turn_count'] > 5
    long_thread = long_thread.values
    long_thread_ids = np.unique(df.loc[long_thread, 'unique_thread_id'].values)
    print(long_thread_ids)
    long_thread = df['unique_thread_id'].isin(long_thread_ids)
    del long_thread_ids
    long_thread_inds = df[long_thread].index
    df.drop(long_thread_inds, inplace=True)
    del long_thread
    del long_thread_inds

    print('Removed threads where more than 6 offers have been made')

    # filter by unique_thread_id, and remove threads where an offer is accepted
    # but there are other offers after it
    prev_ids = len(np.unique(df['unique_thread_id'].values))

    max_turns = df.groupby(['status_id', 'unique_thread_id']).size()
    max_turns_accept = max_turns.xs(1, level='status_id', drop_level=True)
    max_turns_auto = max_turns.xs(9, level='status_id', drop_level=True)
    del max_turns
    auto_inds = max_turns_auto.index
    accept_inds = max_turns_accept.index
    both_inds = np.intersect1d(auto_inds.values, accept_inds.values)
    if both_inds.size > 0:
        print(both_inds)
        both_accept_inds = df[df['unique_thread_id'].isin(both_inds)].index
        df.drop(both_accept_inds, inplace=True)
    df = remove_accept(df, max_turns_accept, both_inds, 1)
    df = remove_accept(df, max_turns_auto, both_inds, 9)

    print('Removed threads that have an accepted offer but not as the last offer')
    cut_ids = len(np.unique(df['unique_thread_id'].values))
    print('Threads had an accept offer entered in the wrong place: %d' %
          (prev_ids - cut_ids))
    print('Total Removed: %d' % (org_ids - cut_ids))
    return df


def add_offrs_counter(df, turn):
    prev_turn = turn - 1
    curr_offr_inds = df[df['offr_type_id'].isin([1, 2])].xs(
        turn, level='turn_count', drop_level=True).index
    prev_offr_inds = [(ind, prev_turn) for ind in curr_offr_inds.values]
    curr_offr_inds = [(ind, turn) for ind in curr_offr_inds.values]
    prev_offrs = df.loc[prev_offr_inds, 'offr_price'].values
    df.loc[curr_offr_inds, 'prev_offr_price'] = prev_offrs
    return df


def add_offrs_init(df):
    df.set_index('turn_count', drop=True, inplace=True)
    byr_turns = df[df['offr_type_id'] == 0].index.values
    slr_turns = df[df['offr_type_id'] == 2].index.values

    if byr_turns.size > 0:
        # sorting seller turn numbers
        slr_turns = np.sort(slr_turns)
        # if there has been at least one seller turn
        if slr_turns.size > 0:
            # find the indices in the slr_turns array where we can insert buyer
            # turns to maintain order
            # Since slr_turns[i] != byr_turns[j] \forall i,j,  these values
            # implicitly correspond to 1 greater than the index in slr_turns
            # of the seller turn before each buyer turn
            insert_points = np.searchsorted(slr_turns, byr_turns)
        else:
            # if there hasn't been any sellers, just make insert points all 0's
            insert_points = np.zeros(len(byr_turns))
        # find the indices of byrs whose insert points corresopnd to 0's
        init_inds = byr_turns[insert_points == 0]
        # boolean array for whether each point is nonzero
        nonzero_inserts = insert_points != 0
        # indices of buyers whose insert points correspond to nonzero values
        other_inds = byr_turns[nonzero_inserts]
        # shrinking insert points to corresopnd only to those with nonzero values
        insert_points = insert_points[nonzero_inserts]
        if init_inds.size > 0:
            # setting all indices of buyers who occur before seller counter offers
            # to have prev offer equal to starting price
            df.loc[init_inds, 'prev_offr_price'] = df.at[0, 'start_price_usd']
        if insert_points.size > 0:
            # incrementing insert_points down 1, so that insert points now give
            # the indices in slr_turns corresponding to the sellers
            # immediately before the nonzero insert points
            insert_points = insert_points - 1
            # grabbing seller indices by indexing by insert_points
            other_sellers = slr_turns[insert_points]
            # extract previous offer values using other_sellers array
            prev_offrs = df.loc[other_sellers, 'offr_price'].values
            df.loc[other_inds, 'prev_offr_price'] = prev_offrs

    count_nan = np.sum(np.isnan(df['prev_offr_price'].values))
    if count_nan > 0:
        print(df[['offr_type_id', 'status_id', 'offr_price', 'start_price_usd',
                  'prev_offr_price', 'resp_offr']])
        raise ValueError('No NANs should escape this phase')
    df.reset_index()
    return df


def prev_offr(df):
    # adding previous offer to initial offer
    prev_offr = pd.Series(np.nan, index=df.index)
    print('Adding initial offers')
    df['prev_offr_price'] = prev_offr
    init_offr_inds = df[df['turn_count'] == 0].index
    start_price = df.loc[init_offr_inds, 'start_price_usd'].copy()
    df.loc[init_offr_inds, 'prev_offr_price'] = start_price.values
    df.set_index(['unique_thread_id', 'turn_count'], inplace=True)
    all_turns = df.index.levels[1]
    for i in range(1, 6):
        if i in all_turns:
            print('Adding turn %d' % (i + 1))
            df = add_offrs_counter(df, i)
    df.reset_index(inplace=True, drop=False)
    df = df.groupby(by='unique_thread_id')

    # to implement if necessary
    count = 0
    group_list = []
    for _, group in df:
        new_group = group.copy()
        new_group = add_offrs_init(new_group)
        if new_group is not None:
            group_list.append(new_group)
            count = count + 1
        if count % 500 == 0:
            gc.collect()
    df = pd.concat(group_list)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--dir', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    print('Reading csv')
    feat_df = pd.read_csv('data/' + subdir + '/' + filename,
                          parse_dates=['src_cre_date', 'auct_start_dt',
                                       'auct_end_dt', 'response_time'],
                          dtype={'unique_thread_id': np.int64})
    print('Done reading')
    feat_df.drop(columns=['item_cndtn_id',
                          'meta_categ_id', 'anon_leaf_categ_id', 'src_cre_dt'], inplace=True)
    feat_df.rename(columns={'Unnamed: 0': 'turn_count'}, inplace=True)
    # Not normalizing for now
    # print('Creating price normalization features')
    # feat_df = norm_price(feat_df)
    print('Creating Date feature')
    feat_df = date_feats(feat_df)
    feat_df = clean_data(feat_df)
    print('Adding previous offer feature')
    feat_df = prev_offr(feat_df)
    print('Writing csv')
    feat_df.to_csv('data/' + subdir + '/' + filename.replace('.csv', '2.csv'))


if __name__ == '__main__':
    main()
