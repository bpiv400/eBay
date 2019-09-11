import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
sys.path.append('repo/')
from constants import *


# parameter values
hidden = np.power(2, np.array(range(4, 11)))
layers = np.array(range(2, 6))
K = np.array(range(2, 9))
lr = np.array(range(-4, -1))

# function to construct dataframe
def create_df(name, cols, l):
	M = cartesian(l)
	idx = pd.Index(range(1, len(M)+1), name='id')
	df = pd.DataFrame(M, index=idx, columns=cols, dtype='int64')
	df.to_csv(EXP_PATH + name + '.csv')
	return df.index[-1]

# function to construct bash file
def create_bash(model, outcome, last):
	f = open('repo/simulator/bash/' + model + '_' + outcome + '.sh', 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -q all.q\n')
	f.write('#$ -l m_mem_free=50G\n')
	f.write('#$ -t 1-%d\n' % last)
	f.write('#$ -N %s_%s\n' % (model, outcome))
	f.write('#$ -o logs/\n\n')
	f.write('python repo/simulator/train.py --model %s --outcome %s --id "$SGE_TASK_ID"'
		% (model, outcome))
	f.close()

# feed-forward non-mixture
ff = create_df('ff', 
	['ff_hidden', 'ff_layers', 'lr'], 
	[hidden, layers, lr])

for outcome in ['bin', 'days', 'hist', 'loc', 'sec']:
	create_bash('arrival', outcome, ff)

# feed-forward mixture
ff_K = create_df('ff_K', 
	['ff_hidden', 'ff_layers', 'K', 'lr'], 
	[hidden, layers, K, lr])

create_bash('arrival', 'sec', ff_K)

# rnn non-mixture
rnn = create_df('rnn', 
	['ff_hidden', 'rnn_hidden', 'ff_layers', 'rnn_layers', 'lr'], 
	[hidden, hidden, layers, layers, lr])

for model in ['byr', 'slr']:
	for outcome in ['accept', 'delay', 'msg', 'nines', 'reject', 'round']:
		create_bash(model, outcome, rnn)

# rnn mixture
rnn_K = create_df('rnn_K', 
	['ff_hidden', 'rnn_hidden', 'ff_layers', 'rnn_layers', 'K', 'lr'], 
	[hidden, hidden, layers, layers, K, lr])

for model in ['byr', 'slr']:
	create_bash(model, 'con', rnn_K)