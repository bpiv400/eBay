# split.py
# Project: First Offer Prediction
# Description: Splits threads into train, test, and pure test, blocks on thread
# Pure test: 15% random sample of data
# Test: 20 % of remaining data
# Train: ALl remaining data (80% of pure test)

# packages
import pandas as pnd
import random
import numpy as np

# renaming imports
read_csv = pnd.read_csv

# extract thread ids
threads = read_csv('data/threads.csv')
unique_threads = set(threads['anon_thread_id'].tolist())

# pure test extraction
pure_test = random.sample(unique_threads, .15 * len(unique_threads))
unique_threads = unique_threads.difference_update(set(pure_test))

# test and train set extraction
test = random.sample(unique_threads, len(unique_threads) * .2)
train = unique_threads.difference_update(set(test))
del unique_threads

# Extracts associated rows from threads dataframe for pure test
ptest_bool = threads['anon_thread_id'].isin(pure_test)
pure_test = threads.loc[ptest_bool.values]
pure_test.to_csv('data/pure_test.csv')
del pure_test

# Extracts rows for test
threads = threads.loc[~ptest_bool.values]
test_bool = threads['anon_thread_id'].isin(test)
test = threads.loc[test_bool.values]
test.to_csv('data/test.csv')
del test

train = threads.loc[~test_bool.values]
train.to_csv('data/train.csv')
