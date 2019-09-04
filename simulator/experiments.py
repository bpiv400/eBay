import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian

# parameter values
hidden = np.power(2, np.array(range(11)))
K = np.array(range(2, 9))

# function to construct dataframe
def create_df(name, cols, l):
	M = cartesian(l)
	idx = pd.Index(range(1, len(M)+1), name='id')
	df = pd.DataFrame(M, index=idx, columns=cols, dtype='int64')
	df.to_csv('experiments/' + name + '.csv')

# non-mixture
create_df('hidden', ['hidden'], [hidden])

# mixture model
create_df('hidden_K', ['hidden', 'K'], [hidden, K])