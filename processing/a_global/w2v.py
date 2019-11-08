import sys, pickle, argparse
import numpy as np, pandas as pd
from compress_pickle import load, dump
from gensim.models import Word2Vec
from constants import *


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
		min_count=1, size=VOCAB_SIZE, workers=0)
	# output dataframe
	leafs = model.wv.vocab.keys()
	output = pd.DataFrame(index=pd.Index([], name='category'), 
		columns=[prefix + str(i) for i in range(VOCAB_SIZE)])
	for leaf in leafs:
		output.loc[leaf] = model.wv.get_vector(leaf)
	return output.sort_index()


if __name__ == '__main__':
	# parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num

	# seller data
	if num == 1:
		print('Preparing seller embeddings')
		L = load(CLEAN_DIR + 'listings.pkl').sort_values(
			['slr', 'start_date', 'lstg'])[['slr', 'cat']]
		L = L.reset_index().set_index('slr').drop('lstg', axis=1).squeeze()

		# run seller model
		print('Training seller embeddings')
		df_slr = run_model(L, 'slr')
		dump(df_slr, W2V_DIR + 'slr.gz')

	# buyer data
	if num == 2:
		print('Preparing buyer embeddings')
		T = load(CLEAN_DIR + 'threads.pkl').sort_values(
			['byr', 'start_time', 'lstg'])['byr']
		T = T.reset_index().drop('thread', axis=1).set_index('byr')

		# join with category
		L = load(CLEAN_DIR + 'listings.pkl')['cat']
		T = T.join(L, on='lstg')['cat']

		# run buyer model
		print('Training buyer embeddings')
		df_byr = run_model(T, 'byr')
		dump(df_byr, W2V_DIR + 'byr.gz')