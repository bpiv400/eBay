import numpy as np, pandas as pd
import sys
from constants import *

# function to construct bash file
def create_bash(model):
	f = open('bash/%s.sh' % model, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -pe openmp 3\n')
	f.write('#$ -N train_%s\n' % model)
	f.write('#$ -o logs/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/models/simulator/train.py --model %s' % model)
	f.close()

# bash files and empty output files
for model in MODELS:
	create_bash(model)