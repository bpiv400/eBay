import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian

# parameter values
dropout = (np.array(range(5)) + 1)
hidden = np.power(2, np.array(range(6)) + 5)
layers = np.power(2, np.array(range(6)) + 1)
K = np.array(range(2, 10))

# function to construct dataframe
def create_df(name, cols, l):
	M = cartesian(l)
	idx = pd.Index(range(1, len(M)+1), name='id')
	df = pd.DataFrame(M, index=idx, columns=cols, dtype='int64')
	df.to_csv('experiments/' + name + '.csv')

# feed-forward
create_df('FF', ['dropout', 'hidden', 'ff_layers'],
	[dropout, hidden, layers])

# feed-forward with mixture model
create_df('FF_K', ['dropout', 'hidden', 'ff_layers', 'K'],
	[dropout, hidden, layers, K])

# LSTM
create_df('LSTM', ['dropout', 'hidden', 'ff_layers', 'lstm_layers'],
	[dropout, hidden, layers, layers])

# LSTM with mixture model
create_df('LSTM_K', ['dropout', 'hidden', 'ff_layers', 'lstm_layers', 'K'],
	[dropout, hidden, layers, layers, K])