import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime as dt
import sys
import os
############################################
# DEPRECATED
# excluding everything else to prevent
# META_DICT = {'1': 'collectibles',
#              '220': 'toys',
#              '237': 'dolls',
#              '260': 'stamps',
#              '267': 'books',
#              '281': 'jewelry',
#              '293': 'computers',
#              '316': 'speciality',
#              '550': 'art',
#              '619': 'musical',
#              '625': 'cameras',
#              '870': 'pottery',
#              '888': 'sports',
#              '1249': 'video_games',
#              '1281': 'pets',
#              '1305': 'tix',
#              '2984': 'baby',
#              '3252': 'travel',
#              '6000': 'motors',
#              '1054': 'real_estate',
#              '1111': 'coins',
#              '1123': 'dvds',
#              '1145': 'clothing',
#              '1170': 'home',
#              '1257': 'business',
#              '1433': 'crafts',
#              '1503': 'cellphones',
#              '2008': 'antiques',
#              '2639': 'health',
#              '4510': 'memerobellia',
#              '5805': 'tablets',
#              '6448': 'fan_shop',
#              '1720': 'coupons'
#              }
##############################################


def genIndicators(all_ids, flag, hold_out=True):
    '''
    Description: Converts anonymized ids into a set of indicator variables and
    removes anonymized id. May be used for meta category id's, leaf ids, condition ids,
    and in general categorical variables encoded in a single column as integers

    Inputs:
        all_ids: a set containing all of the unique anon_leaf_cat_id's in the data set
        flag: 1 character string to append to identify the corresponding indicators
        (and prevent collisions between indicator columns in finalized features)
    Output: dictionary mapping anon_leaf_categ_ids to 1 row data_frames containing the corresponding set of indicators
    '''
    comp_ids = []
    id_strings = []
    # create list of ids as strings (for col names later)
    # create list of ids as ints for index later
    for id in all_ids:
        comp_ids.append(id)
        id_strings.append(str(int(id)) + flag)
    del all_ids
    # convert to nd array for efficient min
    comp_ids = np.array(comp_ids)
    if hold_out:
        # extract min as a string to be used as hold out feature
        hold_out = str(np.amin(comp_ids)) + flag
        # remove hold out id (as string) from list of strings to be used for columns
        id_strings.remove(hold_out)
    # create df of 0's with indices corresponding to all unique id's as ints
    # and columns corresponding to all id's (except hold_out) as strings
    print('Num %s ids : %d' % (flag, len(id_strings)))
    leaf_df = pd.DataFrame(0, index=pd.Index(comp_ids), columns=id_strings)
    # iterate over all id's except the hold out id
    for id_string in id_strings:
        # subset the data frame at the corresponding row index and column name
        # to activate the indicator
        leaf_df.at[int(id_string[0:len(id_string)-1]), id_string] = 1
    return leaf_df

##################################################################################
# DEPRECATED
# NOTE: DEPENDING ON RUN TIME --- CHANGE THIS FUNCTION -- CHANGED
# If this isn't efficient enough, you could equivalently have a function generate a
# dictionary that maps category_id to 1 row data.frames containing the appropriate
# indicator variables --- might just do this
# def genMetaIndicators(row_df):
#     '''
#     Description: Generates a
#         row_df: 1 row pd.DataFrame containing meta_categ_id column
#     Output: pd.DataFrame with new set of indicators and emta_categ_id removed
#     '''
#     # grabs index
#     ind = row_df.index[0]
#
#     category = str(row_df.iloc[0]['meta_categ_id'])
#     all_cats = set(META_DICT.keys())
#     if category in META_DICT:
#         row_df.loc[ind, category] = 1
#     remaining_cats = all_cats.difference(set(category))
#     for cat in remaining_cats:
#         row_df.loc[ind, cat] = 0
#     row_df.drop(columns=['meta_categ_id'], inplace=True)
#     return row_df
######################################################################################

#######################################################################################
# DEPRECEATED
# def sort_dates(df):
#     # sort the thread by date (meaning buyers initial offer will be first)
#     df['src_cre_date'] = pd.to_datetime(df.src_cre_date)
#     df.sort_values(by='src_cre_date', ascending=True,
#                    inplace=True)
#     df.reset_index(drop=True, inplace=True)
#     row_count = len(df.index)
#     # count rows
#     # extract how seller responded to buyer's initial offer
#     response_code = int(df['status_id'].iloc[0])
#     # create new column to encode price of the response offer
#     df['rsp_offr'] = np.nan
#     curr_item_id = df.at[0, 'anon_item_id']
#     list_price = lists.at[curr_item_id, 'start_price_usd']
#     if response_code == 1 or response_code == 9:
#         df.at[0, 'rsp_offr'] = df.at[0, 'offr_price']
#     elif response_code == 7:
#         df.at[0, 'rsp_offr'] = df.at[1, 'offr_price']
#     else:
#         df.at[0, 'rsp_offr'] = list_price
#     if row_count > 1:
#         df = df.iloc[[0]]
#     return df
############################################################################################


def gen_features(df):
    df.sort_values(by='src_cre_date', ascending=True,
                   inplace=True)
    df.reset_index(drop=True, inplace=True)
    early_row = len(df.index)
    curr_item_id = df.at[0, 'anon_item_id']
    listing_row = lists.loc[[curr_item_id]].copy()
    # NB not sure how to use anon_product_id when its missing sometimes, perhaps we can restrict later
    # excluding seller id
    listing_row.drop(columns=['anon_title_code', 'anon_slr_id',
                              'anon_product_id', 'anon_buyer_id', 'ship_time_chosen'], inplace=True)
    # grab leaf, category, and condition id's from listing
    # leaf = listing_row.at[curr_item_id, 'anon_leaf_categ_id']
    categ = listing_row.at[curr_item_id, 'meta_categ_id']
    condition = listing_row.at[curr_item_id, 'item_cndtn_id']
    start_price = listing_row.at[curr_item_id, 'start_price_usd']
    # extract corresponding rows in indicator look up tables
    # leaf_inds = leaf_df.loc[[leaf]]
    categ_inds = cat_df.loc[[categ]]
    if np.isnan(condition):
        cnd_inds = pd.DataFrame(0, index=[-1], columns=cnd_df.columns)
        listing_row.at[curr_item_id, 'item_cndtn_id'] = -1
    else:
        cnd_inds = cnd_df.loc[[condition]]

    # add all indicator columns
    # listing_row.merge(leaf_inds, left_on='anon_leaf_categ_id',
    #                  right_index=True, inplace=True)
    listing_row = listing_row.merge(categ_inds, left_on='meta_categ_id',
                                    right_index=True)
    listing_row = listing_row.merge(cnd_inds, left_on='item_cndtn_id',
                                    right_index=True)
    listing_row = pd.concat([listing_row]*len(df.index), ignore_index=True)
    df = df.join(listing_row, how='right')

    counter_offers = df['status_id'] == 7
    counter_offers = np.nonzero(counter_offers.values)[0]
    if counter_offers.size != 0:
        next_offers = np.add(counter_offers, 1)
        if np.amax(next_offers) < len(df.index):
            next_offer_vals = df.loc[next_offers, 'offr_price']
            if isinstance(next_offer_vals, pd.Series):
                next_offer_vals = next_offer_vals.values
            df.loc[counter_offers, 'resp_offr'] = next_offer_vals
        else:
            print(df[['unique_thread_id', 'anon_thread_id', 'resp_offr', 'offr_price',
                      'start_price_usd', 'offr_type_id', 'status_id']])
            return None
    declined = df['status_id'].isin([0, 2, 6, 8]).values
    if np.sum(declined) > 0:
        seller = df['offr_type_id'] == 2
        seller = seller.values
        # print("seller size " + str(seller.shape))
        buyer = ~seller
        declined_seller = np.nonzero(np.logical_and(declined, seller))[0]
        declined_buyer = np.nonzero(np.logical_and(declined, buyer))[0]
        # print(declined_buyer)
        seller = np.nonzero(seller)[0]
        buyer = np.nonzero(buyer)[0]
        if declined_seller.size > 0:
            # print('has declined seller')
            seller_inds = np.searchsorted(buyer, declined_seller, side='left')
            nonzero_sellers = declined_seller[seller_inds != 0]
            zero_sellers = declined_seller[seller_inds == 0]
            seller_inds = seller_inds[seller_inds != 0]
            # shouldn't need to check size, since none should equal 0, buyer should
            # always occur first a listing
            seller_inds = seller_inds - 1
            prev_buyers = buyer[seller_inds]
            prev_offers = df.loc[prev_buyers, 'offr_price']
            if isinstance(prev_offers, pd.Series):
                prev_offers = prev_offers.values
            df.loc[declined_seller, 'resp_offr'] = prev_offers
        if declined_buyer.size > 0:
            # print('has declined buyer')
            if seller.size != 0:
                buyer_inds = np.searchsorted(
                    seller, declined_buyer, side='left')
                nonzero_buyers = declined_buyer[buyer_inds != 0]
                zero_buyers = declined_buyer[buyer_inds == 0]
                buyer_inds = buyer_inds[buyer_inds != 0]
                # shouldn't need to check size, since none should equal 0, buyer should
                # always occur first a listing
                buyer_inds = buyer_inds - 1
                prev_sellers = seller[buyer_inds]
                prev_offers = df.loc[prev_sellers, 'offr_price']
                if isinstance(prev_offers, pd.Series):
                    prev_offers = prev_offers.values
                if nonzero_buyers.size > 0:
                    df.loc[nonzero_buyers, 'resp_offr'] = prev_offers
                if zero_buyers.size > 0:
                    df.loc[zero_buyers, 'resp_offr'] = start_price
            else:
                df.loc[declined_buyer, 'resp_offr'] = start_price
    late_row = len(df.index)
    rsp = df['resp_offr'].values
    if np.sum(np.isnan(rsp)) != 0:
        print(df[['unique_thread_id', 'resp_offr', 'offr_price',
                  'start_price_usd', 'offr_type_id', 'status_id']])
    if late_row != early_row:
        raise ValueError('Rows have been added')
    return df


def main():
    startTime = dt.now()
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--dir', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    data = pd.read_csv('data/' + subdir + '/' + filename)
    data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    print('Thread file loaded')
    sys.stdout.flush()
    global lists
    lists = pd.read_csv('data/list_chunks/' +
                        filename.replace('.csv', '_lists.csv'))
    # change when ready to run full job
    print(lists.columns)
    lists.set_index('anon_item_id', inplace=True)

    print('Listing file loaded')
    # grabbing relevant indicator values
    # temp: ignore leaves
    # permanent: ignore titles (more than 200000 titles---far too many indicators
    # for this iteration, even leaves are cumbersome (see below, ignored for now)
    condition_values = np.unique(lists['item_cndtn_id'].values)
    condition_values = condition_values[~np.isnan(condition_values)]

    # leaf_values = np.unique(lists['anon_leaf_categ_id'].values)
    categ_values = np.unique(lists['meta_categ_id'].values)
    categ_values = categ_values[~np.isnan(categ_values)]
    print(categ_values)
    print('Indicators grabbed')
    sys.stdout.flush()
    # ignoring leaf indicators for now since there are ~18000 leaves
    # print('Num leaves: ' + str(len(leaf_values)))

    condition_inds = genIndicators(condition_values, 'c', hold_out=False)
    categ_inds = genIndicators(categ_values, 'm', hold_out=True)
    # leaf_inds = genIndicators(leaf_values, 'l')
    print('Indicator tables constructed')
    sys.stdout.flush()

    # pickling each indicator table
    print('Pickling Indicator Tables')
    sys.stdout.flush()
    condition_inds.to_pickle('data/inds/cnd_inds.csv')
    # leaf_inds.to_pickle('data/inds/leaf_inds.csv')
    categ_inds.to_pickle('data/inds/categ_inds.csv')
    print('Indicator tables pickled')
    sys.stdout.flush()
    # convert date of offer creation to datetime
    data['src_cre_date'] = pd.to_datetime(data.src_cre_date)

    # subset data to extract only initial offers, we expect one such for each thread id
    # instance_data = data[data['offr_type_id'] == 0].copy()

    # add response offer price column
    rsp_offer = pd.Series(np.nan, index=data.index)
    data.assign(resp_offr=rsp_offer, inplace=True)

    # extract ids for offers which were accepted
    sys.stdout.flush()
    accepted_bool = data['status_id'].isin(
        [1, 9]).values

    # set accepted id response offers to equal the offer price
    accepted_offer_prices = data.loc[accepted_bool,
                                     'offr_price']
    if isinstance(accepted_offer_prices, pd.Series):
        accepted_offer_prices = accepted_offer_prices.values
    data.loc[accepted_bool,
             'resp_offr'] = accepted_offer_prices

    # get thread ids for threads where seller counter offered initial offer by
    # subsetting instance data

    # TO BE CONTINUED HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Grouping by unique thread id')
    sys.stdout.flush()
    data = data.groupby(by='unique_thread_id')
    global cnd_df
    cnd_df = condition_inds
    global cat_df
    cat_df = categ_inds
    group_list = []
    for _, group in data:
        new_group = group.copy()
        new_group = gen_features(new_group)
        if new_group is not None:
            group_list.append(new_group)
    data = pd.concat(group_list)
    # output=data.apply(gen_features)
    endTime = dt.now()
    print('Total Time: ' + str(endTime - startTime))
    data.to_csv('data/' + subdir + '/' +
                filename.replace('.csv', '') + '_feats.csv')


if __name__ == '__main__':
    main()
