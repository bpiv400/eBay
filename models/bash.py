import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
from constants import *

COLS = ['kl', 'lr']

# hyperparameter values
kl = np.linspace(0.2, 1, num=5)
lr = np.logspace(-1, -3, num=5, base=10)

# create params file
M = cartesian([kl, lr])
idx = pd.Index(range(1, len(M)+1), name='id')
df = pd.DataFrame(M, index=idx, columns=COLS)
df.to_csv(EXPS_DIR + 'params.csv', float_format='%1.3f')

# function to construct bash file
def create_bash(model, T):
	f = open('bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -pe openmp 3\n')
	f.write('#$ -t 1-%d\n' % T)
	f.write('#$ -N train_%s\n' % model)
	f.write('#$ -o logs/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/simulator/train.py --model %s --id "$SGE_TASK_ID"'
		% model)
	f.close()

# bash files and empty output files
for model in MODELS:
	create_bash(model, len(df.index))