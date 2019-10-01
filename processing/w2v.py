import numpy as np, pandas as pd
import sys, pickle
from gensim.models import Word2Vec

INPUT_DIR = 'data/raw/'
OUTPUT_PATH = lambda x: 'data/clean/w2v_' + x + '.csv'
SIZE = 64
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
	L = pd.read_csv('data/clean/listings.csv', 
		usecols=[0,1,2,3,4,7]).sort_values(
		['slr', 'start_date', 'lstg']).drop('start_date', axis=1)

	# convert categories to strings
	for c in ['meta', 'leaf', 'product']:
		L[c] = c[0] + L[c].astype(str)

	# lookup tables
	lstgs = L[['lstg', 'meta', 'leaf', 'product']].set_index(
		'lstg').sort_index()

	# remove one-listing sellers
	L = L.set_index('slr')
	L = L.loc[L['lstg'].groupby(L.index.name).count() > 1]
	L.drop('lstg', axis=1, inplace=True)

	# create category id
	cat_slr = create_category(L)
	del L

	# run seller model
	print('Training seller embeddings')
	df_slr = run_model(cat_slr, 'slr')
	df_slr.to_csv(OUTPUT_PATH('slr'))

	# buyer data
	print('Preparing buyer model')
	T = pd.read_csv('data/clean/threads.csv', 
		usecols=['lstg', 'byr', 'start_time']).sort_values(
		['byr', 'start_time', 'lstg']).drop('start_time', axis=1)

	# remove one-listing buyers
	T = T.set_index('byr').squeeze()
	T = T.loc[T.groupby('byr').count() > 1]

	# join with categories
	T = T.to_frame().join(lstgs, on='lstg').drop('lstg', axis=1).squeeze()

	# create category id
	cat_byr = create_category(T)

	# run buyer model
	print('Training buyer embeddings')
	df_byr = run_model(cat_byr, 'byr')
	df_byr.to_csv(OUTPUT_PATH('byr'))