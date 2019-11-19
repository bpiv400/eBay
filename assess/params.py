import sys, os, pickle
import numpy as np, pandas as pd
from compress_pickle import load
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from constants import *

MODEL = 'msg_slr'

# directory
folder = SUMMARY_DIR + MODEL

# load params file
params = pd.read_csv('%s/params.csv' % folder, 
	index_col=0)[['dropout', 'c', 'b2', 'lr']]

# data
y = load('%s/inputs/train_rl/%s.gz' % (PREFIX, MODEL)).d['y']
baserate = np.mean(y[y > -1])
lnL0 = baserate * np.log(baserate) + (1-baserate) * np.log(1-baserate)

# append output csvs
df = []
n = len(params.index)
for i in range(1,n+1):
	summary = pd.read_csv('%s/%d.csv' % (folder, i))
	summary['id'] = i
	summary.set_index('id', inplace=True)
	df.append(summary)

# concatenate to single dataframe
df = pd.concat(df).sort_values(['id', 'epoch']).dropna()
last = df.lnL_holdout.groupby('id').last()

# does dropout matter?
dropout = [last[params.dropout == x].max() for x in params.dropout.unique()]

# does beta2 matter?
b2 = [last[params.b2 == x].max() for x in params.b2.unique()]

# best parameters
best = params[last == last.min()]

# epoch of max
highest = df.lnL_holdout.groupby('id').max().dropna()
epoch = df.loc[df.lnL_holdout == highest.reindex(df.index), 'epoch'].dropna()

# plot time series of loss
plt.ion()
plt.show()

epochs = df.epoch.max()
ids = np.unique(df.index)
fig, axs = plt.subplots(9, 8, sharex=True, sharey=True)
ct = 0
for idx in ids:
	i = ct // 8
	j = ct % 8
	axs[i, j].plot(range(1, epochs+1), np.repeat(lnL0, epochs))
	axs[i, j].plot(range(1, epochs+1), df.loc[idx, 'lnL_train'])
	axs[i, j].plot(range(1, epochs+1), df.loc[idx, 'lnL_holdout'])
	ct += 1

# y axis range
plt.ylim(-0.7)

