import numpy as np, pandas as pd
import sys, pickle
from gensim.models import Word2Vec

INPUT_DIR = 'data/raw/'
OUTPUT_PATH = 'data/clean/w2v.csv'
DEVPCT = 0.2
SIZE = 256
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
	output = pd.DataFrame(index=pd.Index([], name='leaf'), 
		columns=[prefix + str(i) for i in range(SIZE)])
	for leaf in leafs:
		output.loc[leaf] = model.wv.get_vector(leaf)

	return output.sort_index()


def create_category(df):
	# create collapsed category id
	s = df['product']
	mask = s == 'p0'
	s[mask] = leaf[mask]

	# replace infrequent products with leaf
	ct = s.groupby(s.name).transform('count')
	mask = ct < MIN_COUNT
	s[mask] = leaf[mask]

	# replace infrequent leafs with meta
	ct = s.groupby(s.name).transform('count')
	mask = ct < MIN_COUNT
	s[mask] = meta[mask]

	return s


if __name__ == '__main__':
	# seller data
	print('Loading listings')
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

	# create category id
	cat_slr = create_category(L)

	# run seller model
	print('Training seller embeddings')
	df_slr = run_model(cat_slr, 'slr')
	df_slr.to_csv(OUTPUT_PATH)

	# buyer data
	T = pd.read_csv('data/clean/threads.csv', 
		usecols=[0,2,5]).sort_values(
		['byr', 'start_time', 'lstg']).drop('start_time', axis=1)

	# remove one-listing buyers
	T = T.set_index('byr')
	T = T.loc[T['lstg'].groupby(T.index.name).count() > 1]

	# join with categories
	T = T.join(lstgs, on='lstg').set_index('lstg', append=True)