import pickle
import pandas as pd, numpy as np
from gensim import models as lda

TOPICS = range(2,21)
DIR = './data/lda/'

# test data
bow = pickle.load(open(DIR + 'bow.pkl', 'rb'))['test']

# log-likelihood bound on test data
lnL = {}
for n in TOPICS:
	model = pickle.load(open(DIR + 'm' + str(n) + '.pkl', 'rb'))
	lnL[n] = model.log_perplexity(bow)
	print('%d topics: %1.4f.' % (n, lnL[n]))

# best model
topics = TOPICS[np.argmax(list(lnL.values()))]
model = pickle.load(open(DIR + 'm' + str(topics) + '.pkl', 'rb'))

# save leaf weights
pickle.dump(model.get_topics(), open(DIR + 'weights.pkl', 'wb'))