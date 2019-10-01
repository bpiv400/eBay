import pickle, random
import pandas as pd, numpy as np
from gensim import models as lda

TEST_PCT = .10
SEED = 123123
DIR = './data/lda/'

# read in leaf counts
counts = pd.read_csv(DIR + 'leafcounts.csv', index_col=[0, 1], squeeze=True)

# list of slrs
u = counts.index.unique(level='slr').values

# split into training and test
index = int(u.size * TEST_PCT)
random.seed(SEED)   # set seed
np.random.shuffle(u)
slrs = {'test': u[:index], 'train': u[index:]}

# construct bag of words (i.e., items)
print('Constructing LDA inputs')
bow = {}
for key, val in slrs.items():
	bow[key] = []
	for i in range(len(val)):
		slr = slrs[key][i]
		s = counts.loc[slr]
		bow[key].append(list(zip(s.index, s)))

# TF-IDF statistic
tfidf = lda.TfidfModel(bow['train'])[bow['train']]

# save output
print('Saving LDA inputs')
pickle.dump(bow, open(DIR + 'bow.pkl', 'wb'))
pickle.dump(tfidf, open(DIR + 'tfidf.pkl', 'wb'))



