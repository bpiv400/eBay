import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
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


def genIndicators(all_ids, flag):
    '''
    Description: Converts anonymized ids into a set of indicator variables and
    removes anonymized id. May be used for meta category id's, leaf ids, condition ids,
    and in general categorical variables encoded in a single column as integers

    Inputs:
        all_ids: a set containing all of the unique anon_leaf_cat_id's in the data set
        flag: 1 character string to append to identify the corresponding indicators 
        (and prevent collisions between indicator columns in finalized features)
    Output: dictionary mapping anon_leaf_categ_ids to 1 row data_frames containing the
    corresponding set of indicators
    '''
    comp_ids = []
    id_strings = []
    # create list of ids as strings (for col names later)
    # create list of ids as ints for index later
    for id in all_ids:
        comp_ids.append(id)
        id_strings.append(str(id))
    del all_ids
    # convert to nd array for efficient min
    comp_ids = np.array(comp_ids)
    # extract min as a string to be used as hold out feature
    hold_out = str(np.amin(comp_ids))
    # remove hold out id (as string) from list of strings to be used for columns
    id_strings.remove(hold_out)
    # create df of 0's with indices corresponding to all unique id's as ints
    # and columns corresponding to all id's (except hold_out) as strings
    leaf_df = pd.DataFrame(0, index=pd.Index(comp_ids), columns=id_strings)
    # iterate over all id's except the hold out id
    for id_string in id_strings:
        # subset the data frame at the corresponding row index and column name
        # to activate the indicator
        leaf_df.at[int(id_string), id_string] = 1
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


def gen_features(df, cnd_df=None, cat_df=None, leaf_df=None):
    # sort the thread by date (meaning buyers initial offer will be first)
    df['src_cre_date'] = pd.to_datetime(df.src_cre_date)
    df.sort_values(by='src_cre_date', ascending=True,
                   inplace=True).reset_index(drop=True, inplace=True)
    # count rows
    row_count = len(df.index)
    # extract how seller responded to buyer's initial offer
    response_code = int(df['status_id'].iloc[0])
    # create new column to encode price of the response offer
    df['rsp_offr'] = np.nan
    curr_item_id = df.at[0, 'anon_item_id']
    listing_row = lists.loc[[curr_item_id]]
    # NB not sure how to use anon_product_id when its missing sometimes, perhaps we can restrict later
    # excluding seller id
    listing_row.drop(columns=['anon_title_code', 'anon_slr_id',
                              'anon_item_id', 'anon_product_id', 'anon_buyer_id', 'ship_time_chosen'], inplace=True)

    # grab leaf, category, and condition id's from listing
    leaf = listing_row.at[curr_item_id, 'anon_leaf_categ_id']
    categ = listing_row.at[curr_item_id, 'meta_categ_id']
    condition = listing_row.at[curr_item_id, 'item_cndtn_id']

    # extract corresponding rows in indicator look up tables
    leaf_inds = leaf_df.loc[[leaf]]
    categ_inds = cat_df.loc[[categ]]
    cnd_inds = cnd_df.loc[[condition]]

    # add all indicator columns
    listing_row.merge(leaf_inds, left_on='anon_leaf_categ_id',
                      right_index=True, inplace=True)
    listing_row.merge(categ_inds, left_on='meta_categ_id',
                      right_index=True, inplace=True)
    listing_row.merge(cnd_inds, left_on='item_cndtn_id',
                      right_index=True, inplace=True)

    list_price = listing_row['start_price_usd']
    if response_code == 1 or response_code == 9:
        df.at[0, 'rsp_offr'] = df.at[0, 'offr_price']
    elif response_code == 2 or response_code == 6 or response_code == 8:
        df.at[0, 'rsp_offr'] = list_price
    else:
        df.at[0, 'rsp_offr'] = df.at[1, 'offr_price']
    if row_count > 1:
        df = df.iloc[[0]]
    df.join(listing_row, inplace=True, how='right')
    return df


def par_apply(groupedDf, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in groupedDf])
    return pd.concat(ret_list)


def main():
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    data = pd.read_csv('data/' + filename)
    print('Thread file loaded')
    global lists
    lists = pd.read_csv('data/lists.csv', index_col='anon_item_id')
    print('Listing file loaded')
    # grabbing relevant indicator values
    condition_values = set(data['item_cndtn_id'])
    leaf_values = set(data['anon_leaf_categ_id'])
    categ_values = set(data['meta_categ_id'])
    title_values = set(data['anon_title'])
    print('Indicators grabbed')
    print('Num leaves: ' + str(len(leaf_values)))
    print('Num titles: %d' % len(title_values))

    condition_inds = genIndicators(condition_values, 'c')
    categ_inds = genIndicators(categ_values, 'm')
    leaf_inds = genIndicators(leaf_values, 'l')
    print('Indicator tables constructed')

    # pickling each indicator table
    print('Pickling Indicator Tables')
    condition_inds.to_pickle('data/inds/cnd_inds.csv')
    leaf_inds.to_pickle('data/inds/leaf_inds.csv')
    categ_inds.to_pickle('data/inds/categ_inds.csv')
    print('Indicator tables pickled')

    grouped_data = data.groupby(by='anon_thread_id')
    del data
    print('Threads grouped by thread id')
    applied_gen_features = partial(gen_features, cnd_df=condition_inds,
                                   cat_df=categ_inds,
                                   leaf_df=leaf_inds)

    thread_features = par_apply(grouped_data,  applied_gen_features)
    thread_features.to_csv(
        'data/' + filename.replace('.csv', '') + '_feats.csv')


if __name__ == '__main__':
    main()
