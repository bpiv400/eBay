import numpy as np
import pandas as pd
import sys
import os
import random
import math
import argparse
from datetime import datetime as dt


def main():
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    print('Loading Lists')
    sys.stdout.flush()
    lists = pd.read_csv('data/lists.csv')
    lists.drop_duplicates(subset='anon_item_id', inplace=True)
    lists.set_index('anon_item_id', inplace=True)

    subdir = filename.replace('.csv', '')
    subdir = subdir + '/'
    print('Reading file')
    sys.stdout.flush()
    data = pd.read_csv('data/sorted_tog/' + filename)
    size = data.memory_usage().sum() * math.pow(2, -20)
    n_slice = int(size / 60)
    # group by unique_thread_id
    print('Grouping')
    sys.stdout.flush()
    grouped_data = data.groupby('unique_thread_id')
    n_id = len(grouped_data.groups)
    id_per_slice = int(n_id / n_slice)

    slice_list = []
    slice_dict = {}
    id_counter = 1
    # iterate through groups
    # create a list of dictionaries where each
    # element in the list gives a dictionary of
    # id -> inds for that slice
    print('Creating List of slice dictionaries')
    sys.stdout.flush()
    for thread_id, ids in grouped_data.groups.items():
        if id_counter < id_per_slice:
            id_counter = id_counter + 1
            slice_dict[thread_id] = ids.values
        else:
            slice_list.append(slice_dict)
            slice_dict = {}
            id_counter = 1

    slice_counter = 1
    for slice_dict in slice_list:
        print('Writing slices')
        sys.stdout.flush()
        slice_inds = np.concatenate(list(slice_dict.values()))
        data_slice = data.loc[slice_inds]

        item_ids = data_slice['anon_item_id'].values
        item_ids = np.unique(item_ids)
        list_slice = lists.loc[item_ids]

        data_slice.to_csv(
            'data/' + subdir + filename.replace('.csv', '-') + str(slice_counter) + '.csv')
        list_slice.to_csv('data/list_chunks/' + filename.replace('.csv',
                                                                 '-') + str(slice_counter) + '_lists.csv')
        slice_counter = slice_counter + 1


if __name__ == '__main__':
    main()
