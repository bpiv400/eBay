import pickle, random
import pandas as pd, numpy as np
from gensim import models as lda

TEST_PCT = .10
SEED = 123123
N_TOPICS = range(2, 21)
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
print('Constructing inputs')
bow = {}
for key, val in slrs.items():
	bow[key] = []
	for i in range(len(val)):
		slr = slrs[key][i]
		s = counts.loc[slr]
		bow[key].append(list(zip(s.index, s)))
tfidf = lda.TfidfModel(bow['train'])[bow['train']]	# TF-IDF statistic

# run LDA
print('Running model')
models = {}
lnL = {}
for i in range(len(N_TOPICS)):
	key = N_TOPICS[i]
	models[key] = lda.LdaMulticore(tfidf, num_topics=key, workers=4, eta='auto')
	lnL[key] = models[key].log_perplexity(bow['test'])
	print('%d: %1.4f.' % (key, lnL[key]))

# save output
print('Saving output')
output = {'lnL': lnL, 'models': models}
pickle.dump(output, open(DIR + 'output.pkl', 'wb'))
