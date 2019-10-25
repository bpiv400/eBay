import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys
from constants import *


# parameter values
hidden = np.power(2, np.array(range(7, 11)))
layers = np.array(range(2, 9))

# create params file
M = cartesian([layers, hidden])
idx = pd.Index(range(1, len(M)+1), name='id')
df = pd.DataFrame(M, index=idx, columns=['layers', 'hidden'])
df.to_csv('%s/inputs/params.csv' % PREFIX)

# function to construct bash file
def create_bash(model, T):
	f = open('repo/simulator/bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -q all.q\n')
	f.write('#$ -l m_mem_free=16G\n')
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