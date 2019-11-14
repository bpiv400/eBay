import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
from constants import *

HIDDEN = 1000

# values of max-norm constraint
layers = [2, 4, 8]
dropout = [4, 5, 6]
c = [2, 3, 4, 5]
lr = [-4, -3, -2, -1]
b2 = [-4, -3, -2]

# create params file
M = cartesian([layers, [HIDDEN], dropout, c, b2, lr])
idx = pd.Index(range(1, len(M)+1), name='id')
df = pd.DataFrame(M, index=idx, columns=['layers', 'hidden', 'dropout', 'c', 'b2', 'lr'])
df.to_csv('%s/inputs/params.csv' % PREFIX)

# function to construct bash file
def create_bash(model, T):
	f = open('repo/simulator/bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -l m_mem_free=60G\n')
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