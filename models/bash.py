import numpy as np, pandas as pd
import sys
from constants import *

# function to construct bash file
def create_bash(name):
	f = open('bash/%s.sh' % name, 'w')
	f.write('#!/bin/bash\n')
	f.write('#$ -N %s\n' % name)
	f.write('#$ -o logs/train/\n')
	f.write('#$ -j y\n\n')
	f.write('python repo/models/train.py --name %s' % name)
	f.close()

# bash files and empty output files
for name in MODELS:
	create_bash(name)