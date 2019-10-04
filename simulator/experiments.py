import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys, pickle
sys.path.append('repo/')
from constants import *


# parameter values
hidden = np.power(2, np.array(range(4, 11)))
lstm_hidden = np.power(2, np.array(range(0, 5)))
layers = np.array(range(2, 6))
K = np.array(range(2, 9))

# function to construct dataframe
def create_df(path, cols, l):
	M = cartesian(l)
	idx = pd.Index(range(1, len(M)+1), name='id')
	df = pd.DataFrame(M, index=idx, columns=cols, dtype='int64')
	pickle.dump(df, open(path, 'wb'))
	return df.index[-1]

# function to construct bash file
def create_bash(model, outcome, last):
	f = open('repo/simulator/bash/' + model + '_' + outcome + '.sh', 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -q all.q\n')
	f.write('#$ -l m_mem_free=10G\n')
	f.write('#$ -t 1-%d\n' % last)
	f.write('#$ -N %s_%s\n' % (model, outcome))
	f.write('#$ -o logs/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/simulator/train.py --model %s --outcome %s --id "$SGE_TASK_ID"'
		% (model, outcome))
	f.close()

# path
getPath = lambda model, outcome: 'data/inputs/params/%s_%s.pkl' \
	% (model, outcome)

# days and delay models
for names in [['arrival', 'days'], ['byr', 'delay'], ['slr', 'delay']]:
	path = getPath(*names)
	last = create_df(path, ['ff_hidden', 'ff_layers', 'rnn_hidden'], 
		[hidden, layers, lstm_hidden])
	create_bash(*names, last)

# feed-forward models
for outcome in ['bin', 'hist', 'loc', 'sec']:
	path = getPath('arrival', outcome)
	if outcome == 'sec':
		last = create_df(path, ['ff_hidden', 'ff_layers', 'K'], 
			[hidden, layers, K])
	else:
		last = create_df(path, ['ff_hidden', 'ff_layers'], [hidden, layers])
	create_bash('arrival', outcome, last)

# other recurrent models
for model in ['byr', 'slr']:
	for outcome in ['accept', 'msg', 'nines', 'reject', 'round']:
		path = getPath(model, outcome)
		if outcome == 'con':
			last = create_df(path, 
				['ff_hidden', 'rnn_hidden', 'ff_layers', 'rnn_layers', 'K'], 
				[hidden, hidden, layers, layers, K])
		else:
			last = create_df(path, ['ff_hidden', 'rnn_hidden', 'ff_layers', 'rnn_layers'], 
				[hidden, hidden, layers, layers])
		create_bash(model, outcome, last)
