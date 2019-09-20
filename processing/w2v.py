import numpy as np, pandas as pd
import sys, pickle
from gensim.models import Word2Vec

INPUT_DIR = 'data/raw/'
OUTPUT_PATH = 'data/clean/w2v.csv'
DEVPCT = 0.2
SIZES = np.power(2, range(10))


def run_model(df, leafs):
	# max sentence length
	maxlength = df.groupby(df.index.name).count().max()

	# train and test ids
	u = np.unique(df.index.values)
	N = length(u)
	ids = np.sort(np.random.choice(N, round(N * DEVPCT),
		replace=False))

	# split into holdout and train
	holdout = []
	train = []
	for i in range(N):
		if i in ids:
			holdout.append(df.loc[u[i]].values.tolist())
		else:
			train.append(df.loc[u[i]].values.tolist())

	# word2vec, loop over possible vector sizes
	loss = []
	models = []
	for size in SIZES:
		models.append(Word2Vec(train, sg=1,
			min_count=1, window=maxlength, size=size, workers=7))
		loss.append(models[-1].wv.accuracy(holdout))

	# find optimal vector size
	idx = np.argmin(loss)

	# output dataframe
	output = pd.DataFrame(index=leafs, 
		columns=['s' + str(i) for i in range(1, SIZES[idx]+1)])
	for leaf in leafs:
		output.loc[leaf] = models[idx].wv.get_vector(str(leaf))

	return output


if __name__ == '__main__':
	# seller data
	L = pd.read_csv('data/raw/listings.csv', usecols=[0,3,4]).rename(
		columns=lambda x: x.split('_')[1])

	# lookup table
	items = L[['item', 'leaf']].set_index('item').squeeze()

	# array of leafs
	leafs = np.sort(items.unique())

	# clean seller data
	L = L.drop('item', axis=1).set_index('slr').squeeze().sort_index()
	L = L[L.groupby('slr').count() > 1].astype(str)

	# buyer data
	T = pd.read_csv('data/raw/threads.csv', usecols=[0,2]).rename(
		columns=lambda x: x.split('_')[1])
	T = T.set_index('item').join(items).reset_index(
		drop=True).set_index('byr').squeeze().sort_index()
	T = T[T.groupby('byr').count() > 1].astype(str)

	# run seller and buyer models
	df = run_model(L, leafs, 'slr').join(run_model(T, leafs, 'byr'))

	# save
	df.to_csv(OUTPUT_PATH)