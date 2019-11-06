import numpy as np, pandas as pd
import sys, pickle
from compress_pickle import load, dump
from gensim.models import Word2Vec
from constants import *
from processing.processing_utils import *

SIZE = 32


def run_model(s, prefix):
	# construct sentences
	sentences = []
	for idx in np.unique(s.index.values):
		val = s.loc[idx]
		if isinstance(val, str):
			continue
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


if __name__ == '__main__':
	# seller data
	print('Preparing seller embeddings')
	L = load(CLEAN_DIR + 'listings.gz').sort_values(
		['slr', 'start_date', 'lstg'])[['slr', 'cat']]
	L = L.reset_index().set_index('slr')

	# run seller model
	print('Training seller embeddings')
	df_slr = run_model(L['cat'], 'slr')
	dump(df_slr, CLEAN_DIR + 'w2v_slr.gz')

	# buyer data
	print('Preparing buyer embeddings')
	T = load(CLEAN_DIR + 'threads.gz').sort_values(
		['byr', 'start_time', 'lstg'])['byr']
	T = T.reset_index().drop('thread', axis=1).set_index('byr')

	# join with category
	T = T.join(L.reset_index(drop=True).set_index('lstg'), 
		on='lstg')['cat']

	# run buyer model
	print('Training buyer embeddings')
	df_byr = run_model(T, 'byr')
	dump(df_byr, CLEAN_DIR + 'w2v_byr.gz')