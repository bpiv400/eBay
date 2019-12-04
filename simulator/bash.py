import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
from constants import *

MBSIZE = 128
LAYERS = 8
HIDDEN = 1000
DROPOUT = 5
COLS = ['mbsize', 'layers', 'hidden', 'dropout', 'lr']

# hyperparameter values
lr = [-5, -4, -3, -2]

# create params file
M = cartesian([[MBSIZE], [LAYERS], [HIDDEN], [DROPOUT], lr])
idx = pd.Index(range(1, len(M)+1), name='id')
df = pd.DataFrame(M, index=idx, columns=COLS)
df.to_csv('%s/inputs/params.csv' % PREFIX)

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
