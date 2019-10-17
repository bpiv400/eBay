import numpy as np, pandas as pd
from sklearn.utils.extmath import cartesian
import sys, pickle
sys.path.append('repo/')
from constants import *


# function to construct bash file
def create_bash(model):
	f = open('repo/simulator/bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -q all.q\n')
	f.write('#$ -l m_mem_free=10G\n')
	#f.write('#$ -t 1-%d\n' % len(M))
	f.write('#$ -N %s\n' % model)
	f.write('#$ -o logs/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/simulator/train.py --model %s --id "$SGE_TASK_ID"'
		% model)
	f.close()

# bash files
for model in MODELS:
	create_bash(model)