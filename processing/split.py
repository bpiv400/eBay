# split.py
# Project: First Offer Prediction
# Description: Splits threads into train, test, and pure test, blocks on thread
# Pure test: 15% random sample of data
# Test: 20 % of remaining data
# Train: ALl remaining data (80% of pure test)

# packages
print('This must output')
import random
import numpy as np
import pandas as pnd
import sys

# renaming imports
read_csv = pnd.read_csv
print('Check 1')
# extract thread ids
item_buyer_dict = {}

threads = read_csv('data/threads.csv')

all_itm_ids = threads['anon_item_id']
all_slr_ids = threads['anon_slr_id']
all_buyer_ids = threads['anon_byr_id']
unique_ids = np.zeros(len(all_buyer_ids.values))
print('Generating unique ids')
id_count = 0
index_count = 0
for itm, slr, byr in zip(all_itm_ids.values, all_slr_ids.values, all_buyer_ids.values):
    if itm not in item_buyer_dict:
        byr_slr_dict = {}
        byr_slr_dict[slr] = id_count
        item_buyer_dict[itm] = byr_slr_dict
        unique_ids[index_count] = id_count
        id_count = id_count + 1
    else:
        byr_slr_dict = item_buyer_dict[itm]
        if slr in byr_slr_dict:
            unique_ids[index_count] = byr_slr_dict[slr]
        else:
            byr_slr_dict[slr] = id_count
            unique_ids[index_count] = id_count
            id_count = id_count + 1
    index_count = index_count + 1
    if index_count % 500000 == 0:
        print(index_count)
        sys.stdout.flush()

print('Done Generating Unique Ids')

threads['unique_thread_id'] = unique_ids
unique_threads = set(threads['unique_thread_id'].tolist())
print('Check 2')

# pure test extraction
pure_test = random.sample(unique_threads, k=int(.15 * len(unique_threads)))
unique_threads.difference_update(set(pure_test))
print('Check 3')

# test and train set extraction
test = random.sample(unique_threads, int(len(unique_threads) * .2))
unique_threads.difference_update(set(test))
train = unique_threads
toy = random.sample(train, int(.1 * len(train)))

print('Check 4')
# Extracts associated rows from threads dataframe for pure test
ptest_bool = threads['unique_thread_id'].isin(pure_test)
pure_test = threads.loc[ptest_bool.values]
print('to csv check')
pure_test.to_csv('data/pure_test.csv')
del pure_test
print('Check 5')

# Extracts rows for test
threads = threads.loc[~ptest_bool.values]
test_bool = threads['unique_thread_id'].isin(test)
test = threads.loc[test_bool.values]
test.to_csv('data/test.csv')
del test

train = threads.loc[~test_bool.values]
train.to_csv('data/train.csv')

toy = threads[threads['unique_thread_id'].isin(toy)]
toy.to_csv('data/toy.csv')
print('Check 6')
