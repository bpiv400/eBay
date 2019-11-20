import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
from constants import *

MBSIZE = 128
LAYERS = 8
HIDDEN = 1000
DROPOUT = 5
COLS = ['mbsize', 'layers', 'hidden', 'dropout', 'c', 'b2', 'lr']

# hyperparameter values
c = [2, 3, 4, 5]
lr = [-5, -4, -3, -2]
b2 = [-4, -3, -2]

# create params file
M = cartesian([[MBSIZE], [LAYERS], [HIDDEN], [DROPOUT], c, b2, lr])
idx = pd.Index(range(1, len(M)+1), name='id')
df = pd.DataFrame(M, index=idx, columns=COLS)
df.to_csv('~/weka/eBay/inputs/params.csv')

# function to construct bash file
def create_bash(model, T):
	f = open('bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	#f.write('#$ -pe openmp 2\n')
	f.write('#$ -t 1-%d\n' % T)
	f.write('#$ -N %s\n' % model)
	f.write('#$ -o logs/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/simulator/train.py --model %s --id "$SGE_TASK_ID"'
		% model)
	f.close()

# bash files
for model in MODELS:
	create_bash(model, len(df.index))
