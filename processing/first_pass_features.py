import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

# excluding everything else to prevent
META_DICT = {1: 'collectibles',
             220: 'toys',
             237: 'dolls',
             260: 'stamps',
             267: 'books',
             281: 'jewelry',
             293: 'computers',
             316: 'speciality',
             550: 'art',
             619: 'musical',
             625: 'cameras',
             870: 'pottery'}

parser = argparse.ArgumentParser(
    description='associate threads with all relevant variables')
parser.add_argument('--name', action='store', type=str)
args = parser.parse_args()
filename = args.name

data = pd.read_csv('data/' + filename)
lists = pd.read_csv('data/lists.csv', index_col='anon_item_id')


def gen


def genMetaIndicators(row_df):
    '''
    Description: Converts meta category id into a set of indicator variables
    Input: 1 row pd.DataFrame containing meta_categ_id column
    Output: pd.DataFrame with new set of indicators and emta_categ_id removed
    '''
    ind = row_df.iloc[0]['anon_item_id']
    category = int(row_df.iloc[0]['meta_categ_id'])
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
