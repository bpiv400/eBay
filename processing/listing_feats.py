"""
Adds features to listings dataframe:
-quality indicator
-sale time, if not included
-metacategory id indicator
"""

# modules
import argparse
import pandas as pd
import numpy as np


def gen_indicators(all_ids, flag, hold_out=True):
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


def main():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    # load lstings
    lstg_path = 'data/%s/offers/%s_offers.csv' % (datatype, chunk_name)
    lstgs = pd.read_pickle(lstg_path)
    # enumerate all values of condition indicators
    condition_values = [
        '1000',
        '1500',
        '1750',
        '2000',
        '2500',
        '2750',
        '3000',
        '4000',
        '5000',
        '6000',
        '7000'
    ]
    # enumerate all values of the meta-categories
    categ_vals = [
        '1',
        '220',
        '237',
        '260',
        '267',
        '281',
        '293',
        '316',
        '550',
        '619',
        '625',
        '870',
        '888',
        '1249',
        '1281',
        '1305',
        '2984',
        '3252',
        '6000',
        '1054',
        '1111',
        '1123',
        '1145',
        '1170',
        '1257',
        '1433',
        '1503',
        '2008',
        '2639',
        '4510',
        '5805',
        '6448',
        '1720'
    ]
    # convert category values to integers
    condition_values = [int(val) for val in condition_values]
    categ_vals = [int(val) for val in categ_vals]
    # generate sets of indicators for each 
    condition_inds = gen_indicators(condition_values, 'c', hold_out=False)
    categ_inds = gen_indicators(categ_vals, 'm', hold_out=False)


if __name__ == "__main__":
    main()
