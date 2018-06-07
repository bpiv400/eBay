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

# renaming imports
read_csv = pnd.read_csv
print('Check 1')
# extract thread ids
threads = read_csv('data/threads.csv')
unique_threads = set(threads['anon_thread_id'].tolist())
print('Check 2')

# pure test extraction
pure_test = random.sample(unique_threads, k=(.15 * len(unique_threads)))
unique_threads = unique_threads.difference_update(set(pure_test))
print('Check 3')
# test and train set extraction
test = random.sample(unique_threads, len(unique_threads) * .2)
train = unique_threads.difference_update(set(test))
del unique_threads
print('Check 4')
# Extracts associated rows from threads dataframe for pure test
ptest_bool = threads['anon_thread_id'].isin(pure_test)
pure_test = threads.loc[ptest_bool.values]
print('to csv check')
pure_test.to_csv('data/pure_test.csv')
del pure_test
print('Check 5')

# Extracts rows for test
threads = threads.loc[~ptest_bool.values]
test_bool = threads['anon_thread_id'].isin(test)
test = threads.loc[test_bool.values]
test.to_csv('data/test.csv')
del test

toy = random.sample(train, .1 * len(train))

train = threads.loc[~test_bool.values]
train.to_csv('data/train.csv')

toy = threads[threads['anon_thread_id'].isin(toy)]
toy.to_csv('data/toy.csv')
print('Check 6')
