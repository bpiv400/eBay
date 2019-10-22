import numpy as np, pandas as pd
import sys, pickle
from compress_pickle import load, dump
from gensim.models import Word2Vec
from constants import *
from processing.processing_utils import *

SIZE = 32
MIN_COUNT = 100


def run_model(s, prefix):
	# construct sentences
	sentences = []
	for idx in np.unique(s.index.values):
		sentences.append(s.loc[idx].values.tolist())

	# word2vec
	model = Word2Vec(sentences, sg=1, window=15,
		min_count=1, size=SIZE, workers=7)

	# output dataframe
	leafs = model.wv.vocab.keys()
	output = pd.DataFrame(index=pd.Index([], name='category'), 
		columns=[prefix + str(i) for i in range(SIZE)])
	for leaf in leafs:
		output.loc[leaf] = model.wv.get_vector(leaf)

	return output.sort_index()


def create_category(df):
	# create collapsed category id
	mask = df['product'] == 'p0'
	df.loc[mask, 'product'] = df.loc[mask, 'leaf']
	# replace infrequent products with leaf
	ct = df['product'].groupby(df['product']).transform('count')
	mask = ct < MIN_COUNT
	df.loc[mask, 'product'] = df.loc[mask, 'leaf']
	df.drop('leaf', axis=1, inplace=True)
	# replace infrequent leafs with meta
	ct = df['product'].groupby(df['product']).transform('count')
	mask = ct < MIN_COUNT
	df.loc[mask, 'product'] = df.loc[mask, 'meta']
	df.drop('meta', axis=1, inplace=True)
	return df.squeeze()


if __name__ == '__main__':
	# seller data
	print('Preparing seller model')
	L = load(CLEAN_DIR + 'listings.gz').sort_values(
		['slr', 'start_date', 'lstg'])[['slr', 'meta', 'leaf', 'product']]

	# convert categories to strings
	L = categories_to_string(L)

	# lookup tables
	lstgs = L[['meta', 'leaf', 'product']]

	# remove one-listing sellers
	L = L.reset_index().set_index('slr')
	L = L.loc[L['lstg'].groupby(L.index.name).count() > 1]
	L.drop('lstg', axis=1, inplace=True)

	# create category id
	cat_slr = create_category(L)
	del L

	# run seller model
	print('Training seller embeddings')
	df_slr = run_model(cat_slr, 'slr')
	dump(df_slr, CLEAN_DIR + 'w2v_slr.gz')

	# buyer data
	print('Preparing buyer model')
	T = load(CLEAN_DIR + 'threads.gz').sort_values(
		['byr', 'start_time', 'lstg'])['byr']

	# remove one-listing buyers
	T = T.reset_index().drop('thread', axis=1)
	T = T.set_index('byr').squeeze()
	T = T.loc[T.groupby('byr').count() > 1]

	# join with categories
	T = T.to_frame().join(lstgs, on='lstg').drop('lstg', axis=1)
	del lstgs

	# create category id
	cat_byr = create_category(T)

	# run buyer model
	print('Training buyer embeddings')
	df_byr = run_model(cat_byr, 'byr')
	dump(df_byr, CLEAN_DIR + 'w2v_byr.gz')