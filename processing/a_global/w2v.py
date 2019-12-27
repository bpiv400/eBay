import sys, pickle, argparse
import numpy as np, pandas as pd
from compress_pickle import load, dump
from gensim.models import Word2Vec
from constants import *
from processing.processing_consts import *


def run_model(s):
	# construct sentences
	print('Constructing sentences')
	sentences, max_length = [], 0
	for idx in np.unique(s.index.values):
		val = s.loc[idx]
		if isinstance(val, str):
			continue
		sentence = s.loc[idx].values.tolist()
		sentences.append(sentence)
		max_length = np.maximum(max_length, len(sentence))
	# word2vec
	print('Training model')
	model = Word2Vec(sentences, seed=SEED, sg=1, window=max_length,
		min_count=1, size=VOCAB_SIZE, workers=0)
	# output dataframe
	print('Creating output')
	leafs = model.wv.vocab.keys()
	output = pd.DataFrame(index=pd.Index([], name='category'), 
		columns=[str(i) for i in range(VOCAB_SIZE)])
	for leaf in leafs:
		output.loc[leaf] = model.wv.get_vector(leaf)
	return output.sort_index()


if __name__ == '__main__':
	# parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num-1
	role = ['byr', 'slr'][num]

	# load sentences
	s = pd.read_csv(CLEAN_DIR + 'cat_%s.csv' % role, 
		dtype={role: 'int64', 'cat': str}, index_col=0, squeeze=True)

	# run model
	df = run_model(s).rename(lambda x: role + x, axis=1)

	# save
	dump(df, W2V_DIR + '%s.gz' % role)