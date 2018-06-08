import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

# excluding everything else to prevent
META_DICT = {'1': 'collectibles',
             '220': 'toys',
             '237': 'dolls',
             '260': 'stamps',
             '267': 'books',
             '281': 'jewelry',
             '293': 'computers',
             '316': 'speciality',
             '550': 'art',
             '619': 'musical',
             '625': 'cameras',
             '870': 'pottery',
             '888': 'sports',
             '1249': 'video_games'
             '1281': 'pets',
             '1305': 'tix',
             '2984': 'baby',
             '3252': 'travel',
             '6000': 'motors',
             '1054': 'real_estate'
             '1111': 'coins'
             '1123': 'dvds'
             '1145': 'clothing'
             '1170': 'home',
             '1257': 'business'
             '1433': 'crafts',
             '1503': 'cellphones'
             '2008': 'antiques'
             '2639': 'health',
             '4510': 'memerobellia',
             '5805': 'tablets',
             '6448': 'fan_shop',
             '1720': 'coupons'
             }

parser = argparse.ArgumentParser(
    description='associate threads with all relevant variables')
parser.add_argument('--name', action='store', type=str)
args = parser.parse_args()
filename = args.name

data = pd.read_csv('data/' + filename)
lists = pd.read_csv('data/lists.csv', index_col='anon_item_id')


def genLeafIndicators(all_ids):
    '''
    Description: Converts anonymized leaf ids into a set of indicator variables and
    removes anonymized leaf id. Currently not in use--use of an anonymized feature seems
    epistemologically disingenous (but a system could equivalently made where
    this feature isn't anonymized--what's more, buyers have access to it)
    Inputs:
        all_ids: a set containing all of the unique anon_leaf_cat_id's in the data set
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
    hold_out = str(np.amin(real_ids))
    # remove hold out id (as string) from list of strings to be used for columns
    id_strings.remove(hold_out)
    # create df of 0's with indices corresponding to all unique id's as ints
    # and columns corresponding to all id's (except hold_out) as strings
    leaf_df = pd.DataFrame(0, index=pd.Index(comp_ids), columns=id_strings)
    # iterate over all id's except the hold out id
    for string_id in string_ids:
        # subset the data frame at the corresponding row index and column name
        # to activate the indicator
        leaf_df.at[int(string_id), string_id] = 1
    return leaf_df


# NOTE: DEPENDING ON RUN TIME --- CHANGE THIS FUNCTION
# If this isn't efficient enough, you could equivalently have a function generate a
# dictionary that maps category_id to 1 row data.frames containing the appropriate
# indicator variables --- might just do this
def genMetaIndicators(row_df):
    '''
    Description: Converts meta category id into a set of indicator variables
    Input:
        row_df: 1 row pd.DataFrame containing meta_categ_id column
    Output: pd.DataFrame with new set of indicators and emta_categ_id removed
    '''
    # grabs index
    ind = row_df.index[0]

    category = str(row_df.iloc[0]['meta_categ_id'])
    all_cats = set(META_DICT.keys())
    if category in META_DICT:
        row_df.loc[ind, category] = 1
    remaining_cats = all_cats.difference(set(category))
    for cat in remaining_cats:
        row_df.loc[ind, category] = 0
    row_df.drop(columns=['meta_categ_id'], inplace=True)
    return row_df


def sortTurns(df):
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
    listing_row = lists.loc[[df.at[0, 'anon_item_id']]]
    # NB not sure how to use anon_product_id when its missing sometimes, perhaps we can restrict later
    # excluding seller id
    listing_row.drop(columns=['anon_title_code', 'anon_slr_id',
                              'anon_item_id', 'anon_product_id', 'anon_buyer_id', 'ship_time_chosen'], inplace=True)
    listing_row = genMetaIndicators(listing_row)
    list_price = listing_row['start_price_usd']
    if response_code == 1 or response_code == 9:
        df.at[0, 'rsp_offr'] = df.at[0, 'offr_price']
    elif response_code == 2 or response_code == 6 or response_code == 8:
        df.at[0, 'rsp_offr'] = list_price
    else:
        df.at[0, 'rsp_offr'] = df.at[1, 'offr_price']
    if row_count > 1:
        df = df.iloc[[0]]
    df.join(listing_row, inplace=True)


grouped_data = data.groupby(by='anon_thread_id')


def parApply(groupedDf, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in groupedDf])
    return pd.concat(ret_list)
