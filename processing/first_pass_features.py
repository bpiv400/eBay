import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime as dt
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

# improve
# filter all dates first
# extract all threads that contain a row where type == 0 response_code == 7


def sort_dates(df):
    # sort the thread by date (meaning buyers initial offer will be first)
    df['src_cre_date'] = pd.to_datetime(df.src_cre_date)
    df.sort_values(by='src_cre_date', ascending=True,
                   inplace=True)
    df.reset_index(drop=True, inplace=True)
    row_count = len(df.index)
    # count rows
    # extract how seller responded to buyer's initial offer
    response_code = int(df['status_id'].iloc[0])
    # create new column to encode price of the response offer
    df['rsp_offr'] = np.nan
    curr_item_id = df.at[0, 'anon_item_id']
    list_price = lists.at[curr_item_id, 'start_price_usd']
    if response_code == 1 or response_code == 9:
        df.at[0, 'rsp_offr'] = df.at[0, 'offr_price']
    elif response_code == 7:
        df.at[0, 'rsp_offr'] = df.at[1, 'offr_price']
    else:
        df.at[0, 'rsp_offr'] = list_price
    if row_count > 1:
        df = df.iloc[[0]]
    return df


def gen_features(df, cnd_df=None, cat_df=None, leaf_df=None):
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

    return df.join(listing_row, how='right')


def par_apply(groupedDf, func):
    # with Pool(1) as p:
    #    ret_list = p.map(func, [group for name, group in groupedDf])
    #    ret_list = map(func, [group for name, group in groupedDf])
    ret = groupedDf.apply(func)
    # return pd.concat(ret_list)
    return ret


def sort_counter_offers(df):
    df.sort_values(by='src_cre_date', ascending=True,
                   inplace=True)
    return df['offr_price'].values[0]


def main():
    startTime = dt.now()
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    data = pd.read_csv('data/' + filename)
    print('Thread file loaded')
    global lists
    lists = pd.read_csv('data/toy_lists.csv')
    lists.drop_duplicates(subset='anon_item_id', inplace=True)
    lists.set_index('anon_item_id', inplace=True)
    print('Listing file loaded')
    # grabbing relevant indicator values
    # temp: ignore leaves
    # permanent: ignore titles (more than 200000 titles---far too many indicators
    # for this iteration, even leaves are cumbersome (see below, ignored for now)
    condition_values = np.unique(lists['item_cndtn_id'].values)
    condition_values = condition_values[~np.isnan(condition_values)]
    print(type(condition_values[0]))
    # leaf_values = np.unique(lists['anon_leaf_categ_id'].values)
    categ_values = np.unique(lists['meta_categ_id'].values)
    categ_values = categ_values[~np.isnan(categ_values)]
    print(categ_values)
    print('Indicators grabbed')
    # ignoring leaf indicators for now since there are ~18000 leaves
    # print('Num leaves: ' + str(len(leaf_values)))

    condition_inds = genIndicators(condition_values, 'c', hold_out=False)
    categ_inds = genIndicators(categ_values, 'm', hold_out=True)
    # leaf_inds = genIndicators(leaf_values, 'l')
    print('Indicator tables constructed')

    # pickling each indicator table
    print('Pickling Indicator Tables')
    condition_inds.to_pickle('data/inds/cnd_inds.csv')
    # leaf_inds.to_pickle('data/inds/leaf_inds.csv')
    categ_inds.to_pickle('data/inds/categ_inds.csv')
    print('Indicator tables pickled')

    # convert date of offer creation to datetime
    data['src_cre_date'] = pd.to_datetime(data.src_cre_date)

    # subset data to extract only initial offers, we expect one such for each thread id
    instance_data = data[data['offr_type_id'] == 0]
    # add response offer price column
    instance_data['rsp_offr'] == np.nan
    # extract ids for offers which were declined
    declined_ids = instance_data.loc[instance_data['status_id'].isin(
        [0, 2, 6, 8]), 'anon_thread_id']
    # extract ids for offers which were accepted
    accepted_ids = instance_data.loc[instance_data['status_id'].isin(
        [1, 9]), 'anon_thread_id']
    # set the index of the instance data DataFrame to anon_thread id
    instance_data.set_index('anon_thread_id', inplace=True)
    # set accepted id response offers to equal the offer price
    instance_data.loc[accepted_ids,
                      'rsp_offr'] = instance_data.loc[accepted_ids, 'offr_price']
    # extract the item codes for the items corresponding to the declined offers
    declined_items = instance_data[declined_ids, 'anon_item_id']
    # look these up in the lists DataFrame and extract the corresponding starting price
    declined_prices = lists.loc[declined_items, 'start_price_usd']
    # delete unnecesary data
    del declined_items
    # set response offer price for declined items to equal the list price we just extracted
    instance_data.loc[declined_ids, 'rsp_offr'] = declined_prices

    # get thread ids for threads where seller counter offered initial offer by
    # subsetting instance data
    counter_ids = instance_data[np.isnan(
        instance_data.loc['rsp_offr'].values)].index

    # subset original data to only include counter offers
    counter_offers = data[data['offr_type_id'] == 2]
    del data
    # grab rows with thread id corresponding to instance_data where
    # response offer price was left nan
    counter_offers = counter_offers[counter_offers['anon_thread_id'].isin(
        counter_ids)]
    counter_offers = counter_offers.groupby(by='anon_thread_id')

    # too many leaves to make indicators
    # leaf_inds = None
    print('Threads grouped by thread id')
    # applied_gen_features = partial(gen_features, cnd_df=condition_inds,
    #                                cat_df=categ_inds,
    #                                leaf_df=leaf_inds)
    # thread_features = par_apply(grouped_data, sort_date)
    # thread_features = par_apply(grouped_data,  applied_gen_features)
    counter_offers = par_apply(counter_offers, sort_counter_offers)
    print('done date sorting')
    print(type(counter_offers))
    instance_data[counter_offers.index, 'rsp_offr'] = counter_offers.values

    print("sorted by date successfully")
    endTime = dt.now()
    print('Total Time: ' + str(endTime - startTime))
    instance_data.to_csv(
        'data/' + filename.replace('.csv', '') + '_feats.csv')


if __name__ == '__main__':
    main()
